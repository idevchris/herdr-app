//! HERDR: redraw only what changed, on adapters without a GPU.
//!
//! Without a GPU (remote desktop, virtual machines), Direct3D rasterizes on
//! the CPU (WARP), and a frame costs about as much as the pixels it draws.
//! GPUI draws the whole window every frame, so a spinner step costs as much
//! as a new window. Here the renderer keeps the last frame in a texture (the
//! canvas), compares the new scene with the last one (see `damage.rs`),
//! redraws only the changed rectangles into the canvas, copies to the back
//! buffer what it lacks, and tells DWM which rectangles changed. Nothing here
//! knows the UI: it compares the primitives GPUI hands the renderer.
//!
//! On by default for software adapters; `HERDR_APP_PARTIAL_REDRAW=0` turns
//! it off, `=1` turns it on with a GPU too.

use std::slice;

use anyhow::{Context, Result};
use gpui::*;
use windows::Win32::{
    Foundation::RECT,
    Graphics::{
        Direct3D11::*,
        Dxgi::{Common::*, *},
    },
};

use crate::damage::{self, Damage, Rect};

/// Whether `adapter` gets partial redraws.
pub(crate) fn wanted(adapter: &IDXGIAdapter1) -> bool {
    match std::env::var("HERDR_APP_PARTIAL_REDRAW").as_deref() {
        Ok("0") => false,
        Ok("1") => true,
        _ => unsafe { adapter.GetDesc1() }
            .is_ok_and(|desc| desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 != 0),
    }
}

/// What one frame draws.
pub(crate) struct Pass {
    /// Changed since the last frame: drawn into the canvas, told to DWM.
    pub changed: Damage,
    /// Changed since the back buffer we draw next was last shown: copied
    /// from the canvas.
    pub stale: Damage,
}

impl Pass {
    /// The rectangles to draw, or `None` for the whole window.
    pub fn rects(&self) -> Option<&[Rect]> {
        match &self.changed {
            Damage::Rects(rects) => Some(rects),
            _ => None,
        }
    }
}

pub(crate) struct PartialRedraw {
    tracker: damage::Tracker,
    canvas: Option<(ID3D11Texture2D, ID3D11RenderTargetView)>,
    atlas_removals: u64,
    background: Option<WindowBackgroundAppearance>,
    /// The pixels each primitive of the last scene can touch.
    pub touched: Touched,
    diag: Diag,
}

/// MESSUNG: damage with and without the draw order in the hash, appended
/// every 5 s to `HERDR_APP_DAMAGE_LOG`.
#[derive(Default)]
struct Diag {
    previous: Option<damage::Frame>,
    since: Option<std::time::Instant>,
    frames: u64,
    full: [u64; 2],
    pixels: [f64; 2],
    window: f64,
}

impl Diag {
    fn record(&mut self, ordered: &Damage, unordered: Damage, window: Rect) {
        let Some(path) = std::env::var_os("HERDR_APP_DAMAGE_LOG") else {
            return;
        };
        let since = *self.since.get_or_insert_with(std::time::Instant::now);
        self.frames += 1;
        self.window += window.area() as f64;
        for (i, d) in [ordered, &unordered].into_iter().enumerate() {
            match d {
                Damage::None => {}
                Damage::Full => {
                    self.full[i] += 1;
                    self.pixels[i] += window.area() as f64;
                }
                Damage::Rects(rects) => {
                    self.pixels[i] += rects.iter().map(Rect::area).sum::<i64>() as f64;
                }
            }
        }
        if since.elapsed() < std::time::Duration::from_secs(5) {
            return;
        }
        let line = format!(
            "{}x{}: {} frames; with order {} full, {:.1}% px; without order {} full, {:.1}% px\n",
            window.x1,
            window.y1,
            self.frames,
            self.full[0],
            100.0 * self.pixels[0] / self.window.max(1.0),
            self.full[1],
            100.0 * self.pixels[1] / self.window.max(1.0),
        );
        use std::io::Write as _;
        let _ = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .and_then(|mut f| f.write_all(line.as_bytes()));
        let previous = self.previous.take();
        *self = Self {
            previous,
            ..Default::default()
        };
    }
}

impl PartialRedraw {
    /// `buffer_count`: the swap chain's buffers. Flip-sequential buffers
    /// keep their pixels, so the one we get lacks only what changed in the
    /// `buffer_count - 1` frames since it was shown.
    pub fn new(buffer_count: usize) -> Self {
        Self {
            tracker: damage::Tracker::new(buffer_count - 1),
            canvas: None,
            atlas_removals: 0,
            background: None,
            touched: Touched::default(),
            diag: Diag::default(),
        }
    }

    /// Drops the canvas and the last frames: the next frame draws all.
    /// For resizes, lost devices and anything else that drew elsewhere.
    pub fn forget(&mut self) {
        self.canvas = None;
        self.tracker.invalidate();
    }

    /// The canvas, if a frame was begun since the last [`forget`](Self::forget).
    pub fn canvas_view(&self) -> Option<&ID3D11RenderTargetView> {
        self.canvas.as_ref().map(|(_, view)| view)
    }

    /// Compares `scene` with the last frame. `None`: nothing changed, so
    /// nothing is drawn or presented.
    pub fn begin(
        &mut self,
        device: &ID3D11Device,
        scene: &Scene,
        background: WindowBackgroundAppearance,
        atlas_removals: u64,
        width: u32,
        height: u32,
    ) -> Result<Option<Pass>> {
        if self.canvas.is_none() {
            self.canvas = Some(create_canvas(device, width, height)?);
            self.tracker.invalidate();
        }
        // A freed atlas tile can come back with the same id but other
        // pixels, so equal primitives may no longer look equal.
        if atlas_removals != self.atlas_removals || self.background != Some(background) {
            self.atlas_removals = atlas_removals;
            self.background = Some(background);
            self.tracker.invalidate();
        }
        let mut frame = self.tracker.take_frame();
        collect_frame(scene, &mut frame, &mut self.touched);
        let window = Rect::new(0, 0, width as i32, height as i32);
        let (changed, stale) = self.tracker.next(frame, window);
        let mut unordered = damage::Frame::default();
        collect_unordered(scene, &mut unordered);
        let unordered_damage = match &self.diag.previous {
            Some(previous) if changed != Damage::Full => {
                damage::settle(damage::diff(previous, &unordered), window)
            }
            _ => changed.clone(),
        };
        self.diag.previous = Some(unordered);
        self.diag.record(&changed, unordered_damage, window);
        if changed == Damage::None {
            return Ok(None);
        }
        Ok(Some(Pass { changed, stale }))
    }

    /// Copies from the canvas what `back_buffer` lacks.
    pub fn copy_to(
        &self,
        context: &ID3D11DeviceContext,
        back_buffer: &ID3D11Texture2D,
        pass: &Pass,
    ) -> Result<()> {
        let (canvas, _) = self.canvas.as_ref().context("canvas missing")?;
        match &pass.stale {
            Damage::Rects(rects) => {
                for rect in rects {
                    let region = D3D11_BOX {
                        left: rect.x0 as u32,
                        top: rect.y0 as u32,
                        front: 0,
                        right: rect.x1 as u32,
                        bottom: rect.y1 as u32,
                        back: 1,
                    };
                    unsafe {
                        context.CopySubresourceRegion(
                            back_buffer,
                            0,
                            rect.x0 as u32,
                            rect.y0 as u32,
                            0,
                            canvas,
                            0,
                            Some(&region),
                        )
                    };
                }
            }
            _ => unsafe { context.CopyResource(back_buffer, canvas) },
        }
        Ok(())
    }

    /// Records that `pass` was presented.
    pub fn presented(&mut self, pass: Pass) {
        self.tracker.presented(pass.changed);
    }
}

pub(crate) fn to_rect(rect: &Rect) -> RECT {
    RECT {
        left: rect.x0,
        top: rect.y0,
        right: rect.x1,
        bottom: rect.y1,
    }
}

fn create_canvas(
    device: &ID3D11Device,
    width: u32,
    height: u32,
) -> Result<(ID3D11Texture2D, ID3D11RenderTargetView)> {
    let desc = D3D11_TEXTURE2D_DESC {
        Width: width,
        Height: height,
        MipLevels: 1,
        ArraySize: 1,
        Format: DXGI_FORMAT_B8G8R8A8_UNORM,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        Usage: D3D11_USAGE_DEFAULT,
        BindFlags: D3D11_BIND_RENDER_TARGET.0 as u32,
        CPUAccessFlags: 0,
        MiscFlags: 0,
    };
    let mut texture = None;
    unsafe { device.CreateTexture2D(&desc, None, Some(&mut texture))? };
    let texture = texture.context("creating canvas texture")?;
    let mut view = None;
    unsafe { device.CreateRenderTargetView(&texture, None, Some(&mut view))? };
    Ok((texture, view.context("creating canvas view")?))
}

/// The pixels each primitive of a scene can touch, by kind and index, so a
/// pass over one rectangle draws only the primitives meeting it.
#[derive(Default)]
pub(crate) struct Touched {
    pub shadows: Vec<Rect>,
    pub quads: Vec<Rect>,
    pub paths: Vec<Rect>,
    pub underlines: Vec<Rect>,
    pub monochrome_sprites: Vec<Rect>,
    pub subpixel_sprites: Vec<Rect>,
    pub polychrome_sprites: Vec<Rect>,
}

/// The bytes of a primitive: `repr(C)` without padding, uploaded to the GPU
/// the same way, so equal bytes draw equal pixels.
fn bytes_of<T: Copy>(value: &T) -> &[u8] {
    // SAFETY: `T` is a plain `repr(C)` primitive without padding bytes.
    unsafe { slice::from_raw_parts((value as *const T).cast::<u8>(), size_of::<T>()) }
}

fn cover(bounds: &Bounds<ScaledPixels>, grow: f32) -> Rect {
    Rect::cover(
        bounds.origin.x.0 - grow,
        bounds.origin.y.0 - grow,
        bounds.size.width.0 + 2.0 * grow,
        bounds.size.height.0 + 2.0 * grow,
        1,
    )
}

/// The pixels `bounds` (grown by `grow`) can touch inside `mask`.
fn touched(bounds: &Bounds<ScaledPixels>, grow: f32, mask: &ContentMask<ScaledPixels>) -> Rect {
    cover(bounds, grow).intersect(&cover(&mask.bounds, 0.0))
}

/// A sprite's pixels. A rotated one (a spinning icon) covers its turned corners.
fn sprite_touched(
    bounds: &Bounds<ScaledPixels>,
    transformation: &TransformationMatrix,
    mask: &ContentMask<ScaledPixels>,
) -> Rect {
    if *transformation == TransformationMatrix::unit() {
        return touched(bounds, 0.0, mask);
    }
    let m = transformation.rotation_scale;
    let t = transformation.translation;
    let (x0, y0) = (bounds.origin.x.0, bounds.origin.y.0);
    let (x1, y1) = (x0 + bounds.size.width.0, y0 + bounds.size.height.0);
    let (mut min_x, mut min_y) = (f32::MAX, f32::MAX);
    let (mut max_x, mut max_y) = (f32::MIN, f32::MIN);
    // Rows times the point, as `TransformationMatrix::apply` and the shader do.
    for (x, y) in [(x0, y0), (x1, y0), (x0, y1), (x1, y1)] {
        let px = x * m[0][0] + y * m[0][1] + t[0];
        let py = x * m[1][0] + y * m[1][1] + t[1];
        min_x = min_x.min(px);
        min_y = min_y.min(py);
        max_x = max_x.max(px);
        max_y = max_y.max(py);
    }
    if ![min_x, min_y, max_x, max_y].iter().all(|v| v.is_finite()) {
        return cover(&mask.bounds, 0.0);
    }
    Rect::cover(min_x, min_y, max_x - min_x, max_y - min_y, 1).intersect(&cover(&mask.bounds, 0.0))
}

/// MESSUNG: like `collect_frame`, with every draw order set to 0.
fn collect_unordered(scene: &Scene, frame: &mut damage::Frame) {
    let mut t = Touched::default();
    for quad in &scene.quads {
        let mut q = *quad;
        q.order = 0;
        frame.add(damage::hash_bytes(1, bytes_of(&q)), touched(&quad.bounds, 0.0, &quad.content_mask));
    }
    for shadow in &scene.shadows {
        let mut q = *shadow;
        q.order = 0;
        let rect = touched(&shadow.bounds, 3.0 * shadow.blur_radius.0, &shadow.content_mask)
            .union(&touched(&shadow.element_bounds, 0.0, &shadow.content_mask));
        frame.add(damage::hash_bytes(2, bytes_of(&q)), rect);
    }
    for underline in &scene.underlines {
        let mut q = *underline;
        q.order = 0;
        let rect = touched(&underline.bounds, underline.thickness.0, &underline.content_mask);
        frame.add(damage::hash_bytes(3, bytes_of(&q)), rect);
    }
    for sprite in &scene.monochrome_sprites {
        let mut q = *sprite;
        q.order = 0;
        let rect = sprite_touched(&sprite.bounds, &sprite.transformation, &sprite.content_mask);
        frame.add(damage::hash_bytes(4, bytes_of(&q)), rect);
    }
    for sprite in &scene.subpixel_sprites {
        let mut q = *sprite;
        q.order = 0;
        let rect = sprite_touched(&sprite.bounds, &sprite.transformation, &sprite.content_mask);
        frame.add(damage::hash_bytes(5, bytes_of(&q)), rect);
    }
    for sprite in &scene.polychrome_sprites {
        let mut q = *sprite;
        q.order = 0;
        frame.add(damage::hash_bytes(6, bytes_of(&q)), touched(&sprite.bounds, 0.0, &sprite.content_mask));
    }
    let _ = &mut t;
}

/// Every primitive of `scene`, hashed with the pixels it can touch.
fn collect_frame(scene: &Scene, frame: &mut damage::Frame, t: &mut Touched) {
    for list in [
        &mut t.shadows,
        &mut t.quads,
        &mut t.paths,
        &mut t.underlines,
        &mut t.monochrome_sprites,
        &mut t.subpixel_sprites,
        &mut t.polychrome_sprites,
    ] {
        list.clear();
    }
    for quad in &scene.quads {
        let rect = touched(&quad.bounds, 0.0, &quad.content_mask);
        frame.add(damage::hash_bytes(1, bytes_of(quad)), rect);
        t.quads.push(rect);
    }
    for shadow in &scene.shadows {
        // The shader draws a blurred shadow 3 blur radii past its bounds.
        let rect = touched(&shadow.bounds, 3.0 * shadow.blur_radius.0, &shadow.content_mask)
            .union(&touched(&shadow.element_bounds, 0.0, &shadow.content_mask));
        frame.add(damage::hash_bytes(2, bytes_of(shadow)), rect);
        t.shadows.push(rect);
    }
    for underline in &scene.underlines {
        let rect = touched(&underline.bounds, underline.thickness.0, &underline.content_mask);
        frame.add(damage::hash_bytes(3, bytes_of(underline)), rect);
        t.underlines.push(rect);
    }
    for sprite in &scene.monochrome_sprites {
        let rect = sprite_touched(&sprite.bounds, &sprite.transformation, &sprite.content_mask);
        frame.add(damage::hash_bytes(4, bytes_of(sprite)), rect);
        t.monochrome_sprites.push(rect);
    }
    for sprite in &scene.subpixel_sprites {
        let rect = sprite_touched(&sprite.bounds, &sprite.transformation, &sprite.content_mask);
        frame.add(damage::hash_bytes(5, bytes_of(sprite)), rect);
        t.subpixel_sprites.push(rect);
    }
    for sprite in &scene.polychrome_sprites {
        let rect = touched(&sprite.bounds, 0.0, &sprite.content_mask);
        frame.add(damage::hash_bytes(6, bytes_of(sprite)), rect);
        t.polychrome_sprites.push(rect);
    }
    for path in &scene.paths {
        // SAFETY: `PathVertex<ScaledPixels>` is `repr(C)` f32s, no padding.
        let vertices = unsafe {
            slice::from_raw_parts(
                path.vertices.as_ptr().cast::<u8>(),
                std::mem::size_of_val(path.vertices.as_slice()),
            )
        };
        let mut hash = damage::hash_bytes(7, vertices);
        hash = damage::hash_bytes(hash, bytes_of(&path.color));
        hash = damage::hash_bytes(hash, bytes_of(&path.bounds));
        hash = damage::hash_bytes(hash, bytes_of(&path.content_mask));
        hash = damage::hash_bytes(hash, bytes_of(&path.order));
        let rect = touched(&path.bounds, 0.0, &path.content_mask);
        frame.add(hash, rect);
        t.paths.push(rect);
    }
    // Surfaces are not drawn on Windows.
}
