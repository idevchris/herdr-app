//! HERDR: which parts of the window changed since the last frame.
//!
//! Without a GPU, Direct3D rasterizes on the CPU (WARP), and its cost grows
//! with the pixels drawn. A spinner step changes a few hundred pixels but
//! used to redraw the whole window. The renderer hashes every primitive
//! (content, place and draw order) with the rectangle it can touch; what is
//! only in the old or only in the new frame is damage, and only that is
//! redrawn. Plain std, so the logic is tested without Direct3D.

use std::collections::HashMap;

/// A pixel rectangle, `x0..x1` by `y0..y1`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Rect {
    pub x0: i32,
    pub y0: i32,
    pub x1: i32,
    pub y1: i32,
}

impl Rect {
    pub fn new(x0: i32, y0: i32, x1: i32, y1: i32) -> Self {
        Self { x0, y0, x1, y1 }
    }

    /// The smallest pixel rectangle covering `x, y, width, height`, grown by
    /// `margin` pixels for antialiasing.
    pub fn cover(x: f32, y: f32, width: f32, height: f32, margin: i32) -> Self {
        // `as` saturates (huge content masks), so the margin must too.
        Self {
            x0: (x.floor() as i32).saturating_sub(margin),
            y0: (y.floor() as i32).saturating_sub(margin),
            x1: ((x + width).ceil() as i32).saturating_add(margin),
            y1: ((y + height).ceil() as i32).saturating_add(margin),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.x1 <= self.x0 || self.y1 <= self.y0
    }

    pub fn area(&self) -> i64 {
        if self.is_empty() {
            0
        } else {
            (self.x1 as i64 - self.x0 as i64).saturating_mul(self.y1 as i64 - self.y0 as i64)
        }
    }

    pub fn union(&self, other: &Rect) -> Rect {
        Rect {
            x0: self.x0.min(other.x0),
            y0: self.y0.min(other.y0),
            x1: self.x1.max(other.x1),
            y1: self.y1.max(other.y1),
        }
    }

    pub fn intersect(&self, other: &Rect) -> Rect {
        Rect {
            x0: self.x0.max(other.x0),
            y0: self.y0.max(other.y0),
            x1: self.x1.min(other.x1),
            y1: self.y1.min(other.y1),
        }
    }

    fn touches(&self, other: &Rect) -> bool {
        self.x0 <= other.x1 && other.x0 <= self.x1 && self.y0 <= other.y1 && other.y0 <= self.y1
    }
}

/// What to redraw.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Damage {
    /// Nothing changed: no drawing, no present.
    None,
    /// Everything.
    Full,
    /// Only these rectangles (inside the window, at most [`MAX_RECTS`]).
    Rects(Vec<Rect>),
}

/// One scissor pass per rectangle: Direct3D 11 has one scissor per viewport.
/// Each pass draws only the primitives inside its rectangle (see [`runs`]),
/// so more passes cost little; fewer would merge far-apart spinners into
/// big boxes.
pub(crate) const MAX_RECTS: usize = 16;
/// Above this many raw rectangles, merging pairwise costs more than it saves.
const MAX_RAW_RECTS: usize = 64;
/// Damage covering more than this share of the window redraws all of it.
const FULL_SHARE: f64 = 0.5;

/// Hashes primitive bytes (FNV-1a over 8-byte words, then mixed).
pub(crate) fn hash_bytes(seed: u64, bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64 ^ seed;
    let mut chunks = bytes.chunks_exact(8);
    for chunk in &mut chunks {
        let word = u64::from_le_bytes(chunk.try_into().unwrap());
        hash = (hash ^ word).wrapping_mul(0x0000_0100_0000_01b3);
        hash ^= hash >> 29;
    }
    for &byte in chunks.remainder() {
        hash = (hash ^ byte as u64).wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash ^= hash >> 33;
    hash = hash.wrapping_mul(0xff51_afd7_ed55_8ccd);
    hash ^ (hash >> 33)
}

/// One frame's primitives: hash -> (count, rectangle).
#[derive(Default)]
pub(crate) struct Frame {
    items: HashMap<u64, (u32, Rect)>,
}

impl Frame {
    pub fn clear(&mut self) {
        self.items.clear();
    }

    pub fn add(&mut self, hash: u64, rect: Rect) {
        let entry = self.items.entry(hash).or_insert((0, rect));
        entry.0 += 1;
        entry.1 = entry.1.union(&rect);
    }

    pub fn len(&self) -> usize {
        self.items.len()
    }
}

/// Rectangles of primitives that are in only one of the frames (or in both,
/// a different number of times).
pub(crate) fn diff(old: &Frame, new: &Frame) -> Vec<Rect> {
    let mut rects = Vec::new();
    for (hash, (count, rect)) in &new.items {
        if old.items.get(hash).map(|(c, _)| c) != Some(count) {
            rects.push(*rect);
        }
    }
    for (hash, (_, rect)) in &old.items {
        if !new.items.contains_key(hash) {
            rects.push(*rect);
        }
    }
    rects
}

/// Clips `rects` to `window`, merges them to at most [`MAX_RECTS`], and
/// decides between none, some and all.
pub(crate) fn settle(rects: Vec<Rect>, window: Rect) -> Damage {
    let mut rects: Vec<Rect> = rects
        .into_iter()
        .map(|r| r.intersect(&window))
        .filter(|r| !r.is_empty())
        .collect();
    if rects.is_empty() {
        return Damage::None;
    }
    if rects.len() > MAX_RAW_RECTS {
        let all = rects.iter().skip(1).fold(rects[0], |a, r| a.union(r));
        rects = vec![all];
    }
    // Touching rectangles first, then whichever pair grows the least.
    loop {
        let mut best: Option<(usize, usize, i64)> = None;
        for i in 0..rects.len() {
            for j in i + 1..rects.len() {
                let union = rects[i].union(&rects[j]);
                let growth = if rects[i].touches(&rects[j]) {
                    i64::MIN
                } else {
                    union.area() - rects[i].area() - rects[j].area()
                };
                if best.is_none_or(|(_, _, g)| growth < g) {
                    best = Some((i, j, growth));
                }
            }
        }
        match best {
            Some((i, j, growth)) if growth == i64::MIN || rects.len() > MAX_RECTS => {
                rects[i] = rects[i].union(&rects[j]);
                rects.swap_remove(j);
            }
            _ => break,
        }
    }
    let damaged: i64 = rects.iter().map(Rect::area).sum();
    if damaged as f64 > window.area() as f64 * FULL_SHARE {
        Damage::Full
    } else {
        Damage::Rects(rects)
    }
}

/// The runs of `range` whose rectangle in `touched` meets `clip`, in order:
/// what a scissor pass over `clip` must draw. Primitives outside cannot change
/// a pixel inside, so skipping them keeps the draw order of the rest.
pub(crate) fn runs(
    touched: &[Rect],
    range: std::ops::Range<usize>,
    clip: &Rect,
) -> Vec<std::ops::Range<usize>> {
    let mut out: Vec<std::ops::Range<usize>> = Vec::new();
    for i in range {
        // Unknown (missing) rectangles are drawn, to be safe.
        if touched.get(i).is_some_and(|r| r.intersect(clip).is_empty()) {
            continue;
        }
        match out.last_mut() {
            Some(last) if last.end == i => last.end = i + 1,
            _ => out.push(i..i + 1),
        }
    }
    out
}

/// Rectangles that together cover `a` and `b` (for the buffer-age mode).
pub(crate) fn combine(a: &Damage, b: &Damage, window: Rect) -> Damage {
    match (a, b) {
        (Damage::Full, _) | (_, Damage::Full) => Damage::Full,
        (Damage::None, other) | (other, Damage::None) => other.clone(),
        (Damage::Rects(x), Damage::Rects(y)) => {
            settle(x.iter().chain(y.iter()).copied().collect(), window)
        }
    }
}

/// The previous frame and, for drawing straight into the swap chain, the
/// damage of the frames still sitting in its other buffers.
pub(crate) struct Tracker {
    previous: Option<Frame>,
    spare: Frame,
    /// Damage of the last `history.len()` presented frames, newest first.
    history: Vec<Damage>,
    history_len: usize,
}

impl Tracker {
    /// `history_len`: how many earlier frames a back buffer can be behind
    /// (buffer count - 1); 0 when drawing into a buffer that keeps the last
    /// frame.
    pub fn new(history_len: usize) -> Self {
        Self {
            previous: None,
            spare: Frame::default(),
            history: vec![Damage::Full; history_len],
            history_len,
        }
    }

    /// Forget everything: the next frames redraw all.
    pub fn invalidate(&mut self) {
        self.previous = None;
        self.history = vec![Damage::Full; self.history_len];
    }

    /// An empty frame to fill for [`Tracker::next`].
    pub fn take_frame(&mut self) -> Frame {
        let mut frame = std::mem::take(&mut self.spare);
        frame.clear();
        frame
    }

    /// Compares `frame` with the previous one and keeps it. Returns the
    /// damage since the last frame (what changed on screen) and what must be
    /// drawn into a back buffer that is `history_len` frames old.
    pub fn next(&mut self, frame: Frame, window: Rect) -> (Damage, Damage) {
        let damage = match &self.previous {
            None => Damage::Full,
            Some(previous) => settle(diff(previous, &frame), window),
        };
        if let Some(old) = self.previous.replace(frame) {
            self.spare = old;
        }
        let draw = self
            .history
            .iter()
            .fold(damage.clone(), |acc, d| combine(&acc, d, window));
        (damage, draw)
    }

    /// Records that a frame with `damage` was presented.
    pub fn presented(&mut self, damage: Damage) {
        if self.history_len == 0 {
            return;
        }
        self.history.pop();
        self.history.insert(0, damage);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const WINDOW: Rect = Rect {
        x0: 0,
        y0: 0,
        x1: 1000,
        y1: 800,
    };

    fn frame(items: &[(u64, Rect)]) -> Frame {
        let mut frame = Frame::default();
        for (hash, rect) in items {
            frame.add(*hash, *rect);
        }
        frame
    }

    #[test]
    fn unchanged_frame_has_no_damage() {
        let a = frame(&[(1, Rect::new(0, 0, 10, 10)), (2, Rect::new(20, 20, 30, 30))]);
        let b = frame(&[(2, Rect::new(20, 20, 30, 30)), (1, Rect::new(0, 0, 10, 10))]);
        assert_eq!(settle(diff(&a, &b), WINDOW), Damage::None);
    }

    #[test]
    fn changed_primitive_damages_old_and_new_place() {
        // A spinner glyph moved: both places are redrawn.
        let a = frame(&[(1, Rect::new(0, 0, 10, 10)), (9, Rect::new(500, 0, 600, 10))]);
        let b = frame(&[(2, Rect::new(5, 0, 15, 10)), (9, Rect::new(500, 0, 600, 10))]);
        assert_eq!(
            settle(diff(&a, &b), WINDOW),
            Damage::Rects(vec![Rect::new(0, 0, 15, 10)])
        );
    }

    #[test]
    fn vanished_primitive_is_damage() {
        let a = frame(&[(1, Rect::new(0, 0, 10, 10)), (2, Rect::new(40, 40, 50, 50))]);
        let b = frame(&[(1, Rect::new(0, 0, 10, 10))]);
        assert_eq!(
            settle(diff(&a, &b), WINDOW),
            Damage::Rects(vec![Rect::new(40, 40, 50, 50)])
        );
    }

    #[test]
    fn duplicate_count_matters() {
        // The same translucent quad drawn twice blends twice.
        let a = frame(&[(1, Rect::new(0, 0, 10, 10))]);
        let b = frame(&[(1, Rect::new(0, 0, 10, 10)), (1, Rect::new(0, 0, 10, 10))]);
        assert_eq!(
            settle(diff(&a, &b), WINDOW),
            Damage::Rects(vec![Rect::new(0, 0, 10, 10)])
        );
    }

    #[test]
    fn far_apart_changes_stay_separate_up_to_the_limit() {
        // Spinners far apart: kept apart while there are few, merged beyond.
        let rects: Vec<Rect> = (0..MAX_RECTS as i32 + 4)
            .map(|i| Rect::new(i * 45, i * 38, i * 45 + 24, i * 38 + 24))
            .collect();
        let Damage::Rects(out) = settle(rects[..6].to_vec(), WINDOW) else {
            panic!("expected rects");
        };
        assert_eq!(out.len(), 6);
        let Damage::Rects(out) = settle(rects.clone(), WINDOW) else {
            panic!("expected rects");
        };
        assert_eq!(out.len(), MAX_RECTS);
        for r in &rects {
            assert!(out.iter().any(|o| o.intersect(r) == *r), "{r:?} not covered");
        }
    }

    #[test]
    fn runs_keep_only_primitives_meeting_the_clip_in_order() {
        let touched = [
            Rect::new(0, 0, 10, 10),
            Rect::new(5, 5, 15, 15),
            Rect::new(100, 100, 110, 110),
            Rect::new(0, 0, 50, 50),
            Rect::new(200, 0, 210, 10),
        ];
        let clip = Rect::new(8, 8, 20, 20);
        assert_eq!(runs(&touched, 0..5, &clip), vec![0..2, 3..4]);
        assert_eq!(runs(&touched, 1..3, &clip), vec![1..2]);
        assert!(runs(&touched, 2..3, &clip).is_empty());
        // Past the known rectangles: drawn.
        assert_eq!(runs(&touched, 4..7, &clip), vec![5..7]);
    }

    #[test]
    fn large_damage_redraws_everything() {
        let rects = vec![Rect::new(0, 0, 1000, 500)];
        assert_eq!(settle(rects, WINDOW), Damage::Full);
    }

    #[test]
    fn damage_is_clipped_to_the_window() {
        let rects = vec![Rect::new(-5, -5, 10, 10), Rect::new(2000, 0, 2100, 10)];
        assert_eq!(
            settle(rects, WINDOW),
            Damage::Rects(vec![Rect::new(0, 0, 10, 10)])
        );
    }

    #[test]
    fn many_rects_collapse_to_their_bounds() {
        let rects: Vec<Rect> = (0..100).map(|i| Rect::new(i, 0, i + 1, 1)).collect();
        assert_eq!(
            settle(rects, WINDOW),
            Damage::Rects(vec![Rect::new(0, 0, 100, 1)])
        );
    }

    #[test]
    fn cover_rounds_out_and_adds_margin() {
        assert_eq!(Rect::cover(1.5, 2.2, 3.0, 4.0, 1), Rect::new(0, 1, 6, 8));
    }

    #[test]
    fn huge_bounds_saturate_instead_of_overflowing() {
        let r = Rect::cover(-1e30, -1e30, 2e30, 2e30, 1);
        assert_eq!(r, Rect::new(i32::MIN, i32::MIN, i32::MAX, i32::MAX));
        assert!(r.area() > 0);
        assert_eq!(settle(vec![r], WINDOW), Damage::Full);
    }

    #[test]
    fn hash_sees_every_byte() {
        let a = [1u8; 37];
        let mut b = a;
        b[36] = 2;
        assert_ne!(hash_bytes(0, &a), hash_bytes(0, &b));
        assert_ne!(hash_bytes(0, &a), hash_bytes(1, &a));
    }

    #[test]
    fn tracker_first_frame_is_full_then_only_changes() {
        let mut tracker = Tracker::new(0);
        let mut f = tracker.take_frame();
        f.add(1, Rect::new(0, 0, 10, 10));
        assert_eq!(tracker.next(f, WINDOW).0, Damage::Full);
        let mut f = tracker.take_frame();
        f.add(1, Rect::new(0, 0, 10, 10));
        assert_eq!(tracker.next(f, WINDOW), (Damage::None, Damage::None));
        let mut f = tracker.take_frame();
        f.add(2, Rect::new(0, 0, 10, 10));
        let d = Damage::Rects(vec![Rect::new(0, 0, 10, 10)]);
        assert_eq!(tracker.next(f, WINDOW), (d.clone(), d));
        tracker.invalidate();
        let mut f = tracker.take_frame();
        f.add(2, Rect::new(0, 0, 10, 10));
        assert_eq!(tracker.next(f, WINDOW).0, Damage::Full);
    }

    #[test]
    fn buffer_age_draws_the_damage_of_older_buffers_too() {
        let mut tracker = Tracker::new(2);
        let step = |tracker: &mut Tracker, hash: u64| {
            let mut f = tracker.take_frame();
            f.add(100, Rect::new(500, 500, 510, 510));
            f.add(hash, Rect::new(hash as i32 * 20, 0, hash as i32 * 20 + 10, 10));
            let (damage, draw) = tracker.next(f, WINDOW);
            tracker.presented(damage.clone());
            (damage, draw)
        };
        // The first frames fill buffers that hold nothing yet.
        assert_eq!(step(&mut tracker, 1).1, Damage::Full);
        assert_eq!(step(&mut tracker, 2).1, Damage::Full);
        assert_eq!(step(&mut tracker, 3).1, Damage::Full);
        // Now each buffer is three frames old: redraw the last three changes.
        // Frame 4 lands in frame 1's buffer: frames 2, 3 and 4 changed it.
        let (damage, draw) = step(&mut tracker, 4);
        assert_eq!(bounds(&damage), Rect::new(60, 0, 90, 10));
        assert_eq!(bounds(&draw), Rect::new(20, 0, 90, 10));
    }

    fn bounds(damage: &Damage) -> Rect {
        let Damage::Rects(rects) = damage else {
            panic!("expected rects, got {damage:?}")
        };
        rects.iter().skip(1).fold(rects[0], |a, r| a.union(r))
    }
}
