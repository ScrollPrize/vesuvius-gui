//! Per-sample compositing state machines for ray-marched surface painting.
//!
//! Each `CompositionState` is fed one trilinear sample at a time via
//! `update(value) -> bool`; a `false` return short-circuits the walk
//! (e.g. alpha saturated). `result(n)` collapses the accumulated state
//! to a single u8 to write into the output image.
//!
//! `CompositorRef` is a typed wrapper that lets the cache's fast-path
//! ray walker dispatch concrete state updates without going through a
//! per-sample virtual call. The walker matches once at the top and then
//! runs a monomorphized inner loop per arm.

pub trait CompositionState {
    fn update(&mut self, a: u8) -> bool;
    fn result(&self, num_layers: u32) -> u8;
    fn reset(&mut self);
}

pub struct MaxCompositionState {
    value: u8,
}
impl MaxCompositionState {
    pub fn new() -> Self {
        Self { value: 0 }
    }
}
impl CompositionState for MaxCompositionState {
    fn update(&mut self, a: u8) -> bool {
        self.value = self.value.max(a);
        true
    }
    fn result(&self, _num_layers: u32) -> u8 {
        self.value
    }
    fn reset(&mut self) {
        self.value = 0;
    }
}

pub struct NoCompositionState;
impl CompositionState for NoCompositionState {
    fn update(&mut self, _a: u8) -> bool {
        false
    }
    fn result(&self, _num_layers: u32) -> u8 {
        0
    }
    fn reset(&mut self) {}
}

pub struct AlphaCompositionState {
    min: f32,
    max: f32,
    alpha_cutoff: f32,
    opacity: f32,
    value: f32,
    alpha: f32,
}
impl AlphaCompositionState {
    pub fn new(min: f32, max: f32, alpha_cutoff: f32, opacity: f32) -> Self {
        Self {
            min,
            max,
            alpha_cutoff,
            opacity,
            value: 0.0,
            alpha: 0.0,
        }
    }

    /// `(min, max, alpha_cutoff, opacity)` — read back by `OverlayVolume`'s
    /// dual-volume walk (`Compositor::AlphaOverlay`), which accumulates
    /// outside this state machine but uses the same user-facing parameters.
    pub fn params(&self) -> (f32, f32, f32, f32) {
        (self.min, self.max, self.alpha_cutoff, self.opacity)
    }
}
impl CompositionState for AlphaCompositionState {
    fn update(&mut self, a: u8) -> bool {
        let value = ((a as f32 / 255.0 - self.min) / (self.max - self.min)).clamp(0.0, 1.0);

        if value == 0.0 {
            return true;
        }

        let weight = (1.0 - self.alpha) * (value * self.opacity).min(1.0);
        self.value += weight * value;
        self.alpha += weight;

        self.alpha < self.alpha_cutoff
    }
    fn result(&self, _num_layers: u32) -> u8 {
        (self.value * 255.0).clamp(0.0, 255.0) as u8
    }
    fn reset(&mut self) {
        self.value = 0.0;
        self.alpha = 0.0;
    }
}

pub struct AlphaHeightMapCompositionState {
    min: f32,
    max: f32,
    alpha_cutoff: f32,
    opacity: f32,
    alpha: f32,
    depth: f32,
    weighted_depth: f32,
}
impl AlphaHeightMapCompositionState {
    pub fn new(min: f32, max: f32, alpha_cutoff: f32, opacity: f32) -> Self {
        Self {
            min,
            max,
            alpha_cutoff,
            opacity,
            alpha: 0.0,
            depth: 0.0,
            weighted_depth: 0.0,
        }
    }
}
impl CompositionState for AlphaHeightMapCompositionState {
    fn update(&mut self, a: u8) -> bool {
        let value = ((a as f32 / 255.0 - self.min) / (self.max - self.min)).clamp(0.0, 1.0);

        if value == 0.0 {
            self.depth += 1.0;
            return true;
        }

        let weight = (1.0 - self.alpha) * (value * self.opacity).min(1.0);
        self.alpha += weight;
        self.weighted_depth += weight * self.depth;
        self.depth += 1.0;

        self.alpha < self.alpha_cutoff
    }
    fn result(&self, num_layers: u32) -> u8 {
        (255.0 - self.weighted_depth / self.alpha * 255.0 / num_layers as f32).clamp(0.0, 255.0) as u8
    }
    fn reset(&mut self) {
        self.depth = 0.0;
        self.weighted_depth = 0.0;
        self.alpha = 0.0;
    }
}

/// Typed handle that lets a ray walker dispatch sample updates to a
/// concrete `CompositionState` without going through a virtual call per
/// sample. Each arm carries a `&mut` to the live state; the walker is
/// expected to match once and run an inlined inner loop per arm so the
/// `update` call monomorphizes and folds into the trilerp body.
pub enum CompositorRef<'a> {
    Max(&'a mut MaxCompositionState),
    Alpha(&'a mut AlphaCompositionState),
    HeightMap(&'a mut AlphaHeightMapCompositionState),
    None(&'a mut NoCompositionState),
}

impl<'a> CompositorRef<'a> {
    /// Fallback used by the trait-default per-sample loop in
    /// `VoxelVolume::composite_along_normal`. The fast-path override in
    /// `UnifiedVolume` matches once at the top and never calls this on
    /// the hot path.
    #[inline]
    pub fn update(&mut self, value: u8) -> bool {
        match self {
            CompositorRef::Max(s) => s.update(value),
            CompositorRef::Alpha(s) => s.update(value),
            CompositorRef::HeightMap(s) => s.update(value),
            CompositorRef::None(s) => s.update(value),
        }
    }
}

/// Owning counterpart to `CompositorRef`: holds one concrete
/// composition state, lifetime tied to the local stack frame. Built
/// once per `paint()` so the per-pixel inner loop can reuse the
/// allocation (the f32 accumulators reset cheaply between pixels).
pub enum Compositor {
    Max(MaxCompositionState),
    Alpha(AlphaCompositionState),
    HeightMap(AlphaHeightMapCompositionState),
    None(NoCompositionState),
    /// "Value from base, opacity from overlay" mode. For single-volume walks
    /// this behaves exactly like `Alpha` (the inner state runs the regular
    /// alpha accumulation); `OverlayVolume::composite_color_along_normal`
    /// pattern-matches on this variant to run its dual-volume walk instead,
    /// reusing the state's parameters via `AlphaCompositionState::params`.
    AlphaOverlay(AlphaCompositionState),
    /// "Overlay locates the start, base supplies the walk" mode. For
    /// single-volume walks this behaves exactly like `Alpha`;
    /// `OverlayVolume::composite_color_along_normal` pattern-matches on this
    /// variant to first scan the overlay for the surface onset and then run
    /// the regular alpha walk on the base from there.
    AlphaOverlayStart(AlphaCompositionState),
    /// "Alpha from base × overlay" mode. For single-volume walks this behaves
    /// exactly like `Alpha`; `OverlayVolume::composite_color_along_normal`
    /// pattern-matches on this variant to run a dual-volume walk where each
    /// sample's alpha is the product of both volumes' normalized alphas.
    AlphaOverlayCombined(AlphaCompositionState),
}

impl Compositor {
    #[inline]
    pub fn as_ref_mut(&mut self) -> CompositorRef<'_> {
        match self {
            Compositor::Max(s) => CompositorRef::Max(s),
            Compositor::Alpha(s) => CompositorRef::Alpha(s),
            Compositor::HeightMap(s) => CompositorRef::HeightMap(s),
            Compositor::None(s) => CompositorRef::None(s),
            Compositor::AlphaOverlay(s) => CompositorRef::Alpha(s),
            Compositor::AlphaOverlayStart(s) => CompositorRef::Alpha(s),
            Compositor::AlphaOverlayCombined(s) => CompositorRef::Alpha(s),
        }
    }

    #[inline]
    pub fn reset(&mut self) {
        match self {
            Compositor::Max(s) => s.reset(),
            Compositor::Alpha(s) => s.reset(),
            Compositor::HeightMap(s) => s.reset(),
            Compositor::None(s) => s.reset(),
            Compositor::AlphaOverlay(s) => s.reset(),
            Compositor::AlphaOverlayStart(s) => s.reset(),
            Compositor::AlphaOverlayCombined(s) => s.reset(),
        }
    }

    #[inline]
    pub fn result(&self, num_layers: u32) -> u8 {
        match self {
            Compositor::Max(s) => s.result(num_layers),
            Compositor::Alpha(s) => s.result(num_layers),
            Compositor::HeightMap(s) => s.result(num_layers),
            Compositor::None(s) => s.result(num_layers),
            Compositor::AlphaOverlay(s) => s.result(num_layers),
            Compositor::AlphaOverlayStart(s) => s.result(num_layers),
            Compositor::AlphaOverlayCombined(s) => s.result(num_layers),
        }
    }
}
