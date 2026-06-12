use super::composition::{Compositor, MaxCompositionState};
use super::zarr_paint::{ColorScheme, FourColors};
use super::{DrawingConfig, Image, PaintVolume, Volume, VolumeCons, VoxelPaintVolume, VoxelVolume};
use ecolor::Color32;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Deserialize, serde::Serialize)]
pub enum BlendMode {
    /// `out = lerp(base, color, strength)`. Default; matches historical behavior.
    #[default]
    Alpha,
    /// Luminance-preserving tint. Assumes the base is grayscale (the overlay
    /// pipeline only ever runs on top of a gray buffer). Treats the base's
    /// gray value as the HSL lightness L, takes (hue, saturation) from the
    /// tint color, and outputs `HSL(tint_hue, tint_sat * strength, L_base)`.
    /// Strength=0 leaves the base unchanged; strength=1 paints fully toward
    /// the tint hue without losing brightness. Named "multiply" historically;
    /// strict channel multiply darkened the result, which is not what we want.
    Multiply,
}

#[derive(Debug, Clone, Copy, serde::Deserialize, serde::Serialize)]
pub enum OverlayColoring {
    /// 1→red, 2→green, 3→yellow, 4+→blue; 0 → no paint.
    FourColors {
        alpha: f32,
        #[serde(default)]
        mode: BlendMode,
    },
    /// value 255 → `color`, anything else → no paint. Color stored as [r, g, b]
    /// so the enum can derive serde (ecolor's Color32 lacks serde by default).
    Boolean {
        color: [u8; 3],
        alpha: f32,
        #[serde(default)]
        mode: BlendMode,
    },
    /// 0 → no paint; otherwise color = HSV(hue_deg, 1, 1) with strength scaled
    /// by `value/255 * alpha`. In Alpha mode this means small mask values give
    /// faint, fully-saturated tints rather than dim, semi-blended ones (so the
    /// underlying brightness is preserved instead of being lerped toward black).
    Hue {
        hue_deg: f32,
        alpha: f32,
        #[serde(default)]
        mode: BlendMode,
    },
}

impl OverlayColoring {
    pub fn paint(&self, value: u8) -> Option<(Color32, f32)> {
        match *self {
            OverlayColoring::FourColors { alpha, .. } => {
                if value == 0 {
                    None
                } else {
                    Some((FourColors::get_color(value), alpha))
                }
            }
            OverlayColoring::Boolean { color, alpha, .. } => {
                if value == 255 {
                    Some((Color32::from_rgb(color[0], color[1], color[2]), alpha))
                } else {
                    None
                }
            }
            OverlayColoring::Hue { hue_deg, alpha, .. } => {
                if value == 0 {
                    None
                } else {
                    Some((hsv_to_color32(hue_deg, 1.0, 1.0), (value as f32 / 255.0) * alpha))
                }
            }
        }
    }

    pub fn mode(&self) -> BlendMode {
        match *self {
            OverlayColoring::FourColors { mode, .. }
            | OverlayColoring::Boolean { mode, .. }
            | OverlayColoring::Hue { mode, .. } => mode,
        }
    }
}

impl Default for OverlayColoring {
    fn default() -> Self {
        OverlayColoring::FourColors {
            alpha: 0.4,
            mode: BlendMode::Alpha,
        }
    }
}

fn hsv_to_color32(h_deg: f32, s: f32, v: f32) -> Color32 {
    let h = h_deg.rem_euclid(360.0) / 60.0;
    let c = v * s;
    let x = c * (1.0 - (h % 2.0 - 1.0).abs());
    let (r, g, b) = match h as i32 {
        0 => (c, x, 0.0),
        1 => (x, c, 0.0),
        2 => (0.0, c, x),
        3 => (0.0, x, c),
        4 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };
    let m = v - c;
    Color32::from_rgb(
        ((r + m) * 255.0) as u8,
        ((g + m) * 255.0) as u8,
        ((b + m) * 255.0) as u8,
    )
}

fn apply_blend(base: Color32, color: Color32, strength: f32, mode: BlendMode) -> Color32 {
    match mode {
        BlendMode::Alpha => blend_alpha(base, color, strength),
        BlendMode::Multiply => blend_multiply(base, color, strength),
    }
}

fn blend_alpha(base: Color32, value: Color32, alpha: f32) -> Color32 {
    Color32::from_rgba_unmultiplied(
        (base.r() as f32 * (1.0 - alpha) + value.r() as f32 * alpha) as u8,
        (base.g() as f32 * (1.0 - alpha) + value.g() as f32 * alpha) as u8,
        (base.b() as f32 * (1.0 - alpha) + value.b() as f32 * alpha) as u8,
        (base.a() as f32 * (1.0 - alpha) + value.a() as f32 * alpha) as u8,
    )
}

fn blend_multiply(base: Color32, color: Color32, strength: f32) -> Color32 {
    let s = strength.clamp(0.0, 1.0);
    let (h, s_color) = rgb_to_hue_sat(color);
    // base is grayscale (r==g==b); use its gray value as HSL lightness.
    let l = base.r() as f32 / 255.0;
    let (r, g, b) = hsl_to_rgb(h, s_color * s, l);
    Color32::from_rgba_unmultiplied(
        (r * 255.0).clamp(0.0, 255.0) as u8,
        (g * 255.0).clamp(0.0, 255.0) as u8,
        (b * 255.0).clamp(0.0, 255.0) as u8,
        base.a(),
    )
}

fn rgb_to_hue_sat(color: Color32) -> (f32, f32) {
    let r = color.r() as f32 / 255.0;
    let g = color.g() as f32 / 255.0;
    let b = color.b() as f32 / 255.0;
    let max = r.max(g).max(b);
    let min = r.min(g).min(b);
    let delta = max - min;
    if delta < 1e-6 {
        return (0.0, 0.0);
    }
    let l = (max + min) * 0.5;
    let s = if l > 0.5 {
        delta / (2.0 - max - min)
    } else {
        delta / (max + min)
    };
    let h_sector = if max == r {
        ((g - b) / delta).rem_euclid(6.0)
    } else if max == g {
        (b - r) / delta + 2.0
    } else {
        (r - g) / delta + 4.0
    };
    (h_sector * 60.0, s)
}

fn hsl_to_rgb(h_deg: f32, s: f32, l: f32) -> (f32, f32, f32) {
    let c = (1.0 - (2.0 * l - 1.0).abs()) * s;
    let h = h_deg.rem_euclid(360.0) / 60.0;
    let x = c * (1.0 - (h.rem_euclid(2.0) - 1.0).abs());
    let (r1, g1, b1) = match h as i32 {
        0 => (c, x, 0.0),
        1 => (x, c, 0.0),
        2 => (0.0, c, x),
        3 => (0.0, x, c),
        4 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };
    let m = l - c * 0.5;
    (r1 + m, g1 + m, b1 + m)
}

/// Outer overlay for the orthographic XY/XZ/YZ panes: wraps a `Volume` (raw zarr
/// or OME-zarr) and paints colored, alpha-blended pixels on top of whatever is
/// already in the buffer.
pub struct OverlayPaintVolume {
    inner: Volume,
    coloring: OverlayColoring,
}

impl OverlayPaintVolume {
    pub fn new(inner: Volume, coloring: OverlayColoring) -> Self {
        Self { inner, coloring }
    }
}

impl PaintVolume for OverlayPaintVolume {
    /// Render the overlay by reusing the inner volume's `paint` fast path:
    /// paint the inner into a scratch `Image` (which holds the raw label u8
    /// in `Color32::from_gray(v)`), then walk pixels once and blend the
    /// colored result into `buffer`. This avoids the per-pixel
    /// `VoxelVolume::get` lookup that the previous implementation did and
    /// lets cache-backed overlays share the chunk-aware paint loop in
    /// `UnifiedVolume::paint`.
    fn paint(
        &self,
        xyz: [i32; 3],
        u_coord: usize,
        v_coord: usize,
        plane_coord: usize,
        width: usize,
        height: usize,
        sfactor: u8,
        paint_zoom: u8,
        _config: &DrawingConfig,
        buffer: &mut Image,
    ) {
        // Render the inner into a scratch buffer. `from_gray(0)` is the
        // "no label" sentinel — every coloring mode maps value 0 to `None`,
        // so any pixel the inner doesn't touch (chunk not resident yet,
        // out-of-bounds, etc.) safely leaves the main buffer untouched.
        let mut scratch = Image::new_from_color(width, height, Color32::from_gray(0));
        // Use a default config so filters/quantization meant for grayscale
        // CT data don't mangle discrete label values, and no debug overlay
        // gets baked into the labels.
        let inner_config = DrawingConfig::default();
        self.inner.paint(
            xyz,
            u_coord,
            v_coord,
            plane_coord,
            width,
            height,
            sfactor,
            paint_zoom,
            &inner_config,
            &mut scratch,
        );

        let mode = self.coloring.mode();
        for im_v in 0..height {
            let scratch_row = im_v * width;
            let buffer_row = im_v * buffer.width;
            for im_u in 0..width {
                let v = scratch.data[scratch_row + im_u].r();
                if let Some((color, strength)) = self.coloring.paint(v) {
                    let pos = buffer_row + im_u;
                    buffer.data[pos] = apply_blend(buffer.data[pos], color, strength, mode);
                }
            }
        }
    }

    fn shared(&self) -> VolumeCons {
        let inner_cons = self.inner.shared();
        let coloring = self.coloring;
        Box::new(move || OverlayPaintVolume::new(inner_cons(), coloring).into_volume())
    }
}

impl VoxelVolume for OverlayPaintVolume {
    fn get(&self, xyz: [f64; 3], downsampling: i32) -> u8 {
        self.inner.get(xyz, downsampling)
    }
    fn get_interpolated(&self, xyz: [f64; 3], downsampling: i32) -> u8 {
        self.inner.get_interpolated(xyz, downsampling)
    }
    fn reset_for_painting(&self) {
        self.inner.reset_for_painting();
    }
}

/// Inner overlay for the segment/UV pane: wraps a grayscale base + a labeled
/// overlay. `get_color` returns the blended color; `get` keeps grayscale
/// semantics for any consumer that still reads u8 (e.g. ObjVolume composite
/// modes).
pub struct OverlayVolume {
    base: Volume,
    overlay: Volume,
    coloring: OverlayColoring,
}

impl OverlayVolume {
    pub fn new(base: Volume, overlay: Volume, coloring: OverlayColoring) -> Self {
        Self {
            base,
            overlay,
            coloring,
        }
    }
}

impl VoxelVolume for OverlayVolume {
    fn get(&self, xyz: [f64; 3], downsampling: i32) -> u8 {
        self.base.get(xyz, downsampling)
    }
    fn get_interpolated(&self, xyz: [f64; 3], downsampling: i32) -> u8 {
        self.base.get_interpolated(xyz, downsampling)
    }
    fn reset_for_painting(&self) {
        self.base.reset_for_painting();
        self.overlay.reset_for_painting();
    }
    fn get_color(&self, xyz: [f64; 3], downsampling: i32) -> Color32 {
        let base_color = self.base.get_color(xyz, downsampling);
        let lab = self.overlay.get(xyz, downsampling);
        match self.coloring.paint(lab) {
            Some((c, s)) => apply_blend(base_color, c, s, self.coloring.mode()),
            None => base_color,
        }
    }
    fn get_color_interpolated(&self, xyz: [f64; 3], downsampling: i32) -> Color32 {
        let base_color = self.base.get_color_interpolated(xyz, downsampling);
        let lab = self.overlay.get_interpolated(xyz, downsampling);
        match self.coloring.paint(lab) {
            Some((c, s)) => apply_blend(base_color, c, s, self.coloring.mode()),
            None => base_color,
        }
    }

    fn touch_aabb(&self, min: [f64; 3], max: [f64; 3], downsampling: i32) {
        self.base.touch_aabb(min, max, downsampling);
        self.overlay.touch_aabb(min, max, downsampling);
    }

    /// Composite the base and the overlay separately, both through the
    /// cache fast path, then blend the two u8 results into a Color32 via
    /// `OverlayColoring`. The base walk reuses the caller's compositor so
    /// the user's selected compositing mode (Max/Alpha/HeightMap) applies
    /// to brightness as expected. The overlay is a discrete mask volume,
    /// so the overlay walk uses its own `Max` compositor — the strongest
    /// label sample along the ray decides the tint.
    ///
    /// The overlay-aware alpha modes are handled specially instead of the
    /// composite-then-tint blend; all of them sample the volumes via the
    /// fast `gather_along_normal`:
    /// - `AlphaOverlay`: opacity entirely from the overlay, raw base value
    ///   shown — the overlay decides *where* along the ray is visible, the
    ///   base decides *what* is shown.
    /// - `AlphaOverlayStart`: the overlay only locates the front-most
    ///   significant sample; from there the regular alpha walk runs on the
    ///   base. Never-significant rays fall back to the full regular walk.
    /// - `AlphaOverlayCombined`: regular alpha walk on the base, but each
    ///   sample's alpha is scaled by the raw overlay confidence — a
    ///   continuous mask that suppresses non-ink signal sitting on top of
    ///   ink without changing the look where the overlay saturates.
    fn composite_color_along_normal(
        &self,
        base: [f64; 3],
        dir: [f64; 3],
        w_lo: f64,
        w_hi: f64,
        downsampling: i32,
        compositor: &mut Compositor,
        num_layers: u32,
    ) -> Color32 {
        #[derive(Clone, Copy)]
        enum OverlayWalk {
            Opacity,
            Start,
            Combined,
        }
        // Pull the parameters out first so the borrow on the compositor ends
        // (the Start walk reuses it for the base walk below).
        let special = match &*compositor {
            Compositor::AlphaOverlay(s) => Some((OverlayWalk::Opacity, s.params())),
            Compositor::AlphaOverlayStart(s) => Some((OverlayWalk::Start, s.params())),
            Compositor::AlphaOverlayCombined(s) => Some((OverlayWalk::Combined, s.params())),
            _ => None,
        };
        if let Some((walk, (a_min, a_max, a_cutoff, a_opacity))) = special {
            // Stack scratch: layers_in_front/behind are u8, so the segment
            // walk never exceeds 511 samples.
            const MAX_SAMPLES: usize = 512;
            let n = ((w_hi - w_lo).max(0.0) as usize).min(MAX_SAMPLES);
            if n == 0 {
                return Color32::BLACK;
            }
            let start = [base[0] + w_lo * dir[0], base[1] + w_lo * dir[1], base[2] + w_lo * dir[2]];
            let mut alpha_buf = [0u8; MAX_SAMPLES];
            self.overlay.gather_along_normal(start, dir, downsampling, &mut alpha_buf[..n]);

            if let OverlayWalk::Start = walk {
                // First sample (front-to-back) where the overlay is
                // significant, i.e. its normalized alpha is > 0 (raw value
                // above the Alpha Min slider). If the overlay never fires
                // along the ray, run the full regular walk instead.
                let onset = alpha_buf[..n]
                    .iter()
                    .position(|&v| v as f32 / 255.0 > a_min)
                    .unwrap_or(0);
                compositor.reset();
                self.base.composite_along_normal(
                    base,
                    dir,
                    w_lo + onset as f64,
                    w_hi,
                    downsampling,
                    &mut compositor.as_ref_mut(),
                );
                return Color32::from_gray(compositor.result(num_layers));
            }

            let mut value_buf = [0u8; MAX_SAMPLES];
            self.base.gather_along_normal(start, dir, downsampling, &mut value_buf[..n]);

            let mut acc_v = 0.0f32;
            let mut acc_a = 0.0f32;
            for k in 0..n {
                let (a, v) = match walk {
                    // Opacity entirely from the overlay (normalized by the
                    // alpha sliders); raw base value shown.
                    OverlayWalk::Opacity => {
                        let a_o = ((alpha_buf[k] as f32 / 255.0 - a_min) / (a_max - a_min)).clamp(0.0, 1.0);
                        (a_o, value_buf[k] as f32 / 255.0)
                    }
                    // Regular alpha on the base, scaled by the raw overlay
                    // confidence. Identical to `Alpha` where the overlay is
                    // 255; contributes nothing where it is 0 — so bright
                    // non-ink material in front of ink does not occlude it.
                    OverlayWalk::Combined => {
                        let a_b = ((value_buf[k] as f32 / 255.0 - a_min) / (a_max - a_min)).clamp(0.0, 1.0);
                        let a_o = alpha_buf[k] as f32 / 255.0;
                        (a_b * a_o, a_b)
                    }
                    OverlayWalk::Start => unreachable!(),
                };
                if a == 0.0 {
                    continue;
                }
                let weight = (1.0 - acc_a) * (a * a_opacity).min(1.0);
                acc_v += weight * v;
                acc_a += weight;
                if acc_a >= a_cutoff {
                    break;
                }
            }
            return Color32::from_gray((acc_v * 255.0).clamp(0.0, 255.0) as u8);
        }

        compositor.reset();
        self.base.composite_along_normal(
            base,
            dir,
            w_lo,
            w_hi,
            downsampling,
            &mut compositor.as_ref_mut(),
        );
        let base_color = Color32::from_gray(compositor.result(num_layers));

        let mut overlay_comp = Compositor::Max(MaxCompositionState::new());
        self.overlay.composite_along_normal(
            base,
            dir,
            w_lo,
            w_hi,
            downsampling,
            &mut overlay_comp.as_ref_mut(),
        );
        let overlay_val = overlay_comp.result(num_layers);

        match self.coloring.paint(overlay_val) {
            Some((c, s)) => apply_blend(base_color, c, s, self.coloring.mode()),
            None => base_color,
        }
    }
}

impl PaintVolume for OverlayVolume {
    fn paint(
        &self,
        xyz: [i32; 3],
        u_coord: usize,
        v_coord: usize,
        plane_coord: usize,
        width: usize,
        height: usize,
        sfactor: u8,
        paint_zoom: u8,
        config: &DrawingConfig,
        buffer: &mut Image,
    ) {
        self.base.paint(
            xyz,
            u_coord,
            v_coord,
            plane_coord,
            width,
            height,
            sfactor,
            paint_zoom,
            config,
            buffer,
        );
        OverlayPaintVolume::new(self.overlay.clone(), self.coloring).paint(
            xyz,
            u_coord,
            v_coord,
            plane_coord,
            width,
            height,
            sfactor,
            paint_zoom,
            config,
            buffer,
        );
    }

    fn shared(&self) -> VolumeCons {
        let base_cons = self.base.shared();
        let overlay_cons = self.overlay.shared();
        let coloring = self.coloring;
        Box::new(move || OverlayVolume::new(base_cons(), overlay_cons(), coloring).into_volume())
    }
}

#[cfg(test)]
mod tests {
    use super::super::composition::AlphaCompositionState;
    use super::*;

    struct ConstVolume(u8);
    impl VoxelVolume for ConstVolume {
        fn get(&self, _xyz: [f64; 3], _downsampling: i32) -> u8 {
            self.0
        }
    }
    impl PaintVolume for ConstVolume {
        fn paint(
            &self,
            _xyz: [i32; 3],
            _u_coord: usize,
            _v_coord: usize,
            _plane_coord: usize,
            _width: usize,
            _height: usize,
            _sfactor: u8,
            _paint_zoom: u8,
            _config: &DrawingConfig,
            _buffer: &mut Image,
        ) {
        }
        fn shared(&self) -> VolumeCons {
            let v = self.0;
            Box::new(move || ConstVolume(v).into_volume())
        }
    }

    /// Opaque (255) for x >= 5, transparent (0) before.
    struct StepVolume;
    impl VoxelVolume for StepVolume {
        fn get(&self, xyz: [f64; 3], _downsampling: i32) -> u8 {
            if xyz[0] >= 5.0 {
                255
            } else {
                0
            }
        }
    }
    impl PaintVolume for StepVolume {
        fn paint(
            &self,
            _xyz: [i32; 3],
            _u_coord: usize,
            _v_coord: usize,
            _plane_coord: usize,
            _width: usize,
            _height: usize,
            _sfactor: u8,
            _paint_zoom: u8,
            _config: &DrawingConfig,
            _buffer: &mut Image,
        ) {
        }
        fn shared(&self) -> VolumeCons {
            Box::new(|| StepVolume.into_volume())
        }
    }

    /// Base whose value grows along x: v = x * 20 (clamped to u8).
    struct GradientVolume;
    impl VoxelVolume for GradientVolume {
        fn get(&self, xyz: [f64; 3], _downsampling: i32) -> u8 {
            (xyz[0] * 20.0).clamp(0.0, 255.0) as u8
        }
    }
    impl PaintVolume for GradientVolume {
        fn paint(
            &self,
            _xyz: [i32; 3],
            _u_coord: usize,
            _v_coord: usize,
            _plane_coord: usize,
            _width: usize,
            _height: usize,
            _sfactor: u8,
            _paint_zoom: u8,
            _config: &DrawingConfig,
            _buffer: &mut Image,
        ) {
        }
        fn shared(&self) -> VolumeCons {
            Box::new(|| GradientVolume.into_volume())
        }
    }

    // min=0, max=1 (raw value as alpha), cutoff=0.95, opacity=1.
    fn alpha_state() -> AlphaCompositionState {
        AlphaCompositionState::new(0.0, 1.0, 0.95, 1.0)
    }

    fn alpha_overlay_compositor() -> Compositor {
        Compositor::AlphaOverlay(alpha_state())
    }

    #[test]
    fn dual_walk_takes_value_from_base_where_overlay_is_opaque() {
        let vol = OverlayVolume::new(
            ConstVolume(200).into_volume(),
            StepVolume.into_volume(),
            OverlayColoring::default(),
        );
        let mut comp = alpha_overlay_compositor();
        // Walk along +x from the origin; the overlay turns opaque at x=5, so
        // the first contributing sample fully saturates alpha with the base
        // value 200.
        let color = vol.composite_color_along_normal([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], 0.0, 10.0, 1, &mut comp, 10);
        assert!((color.r() as i32 - 200).abs() <= 1, "got {:?}", color);
        assert_eq!(color.r(), color.g());
        assert_eq!(color.r(), color.b());
    }

    #[test]
    fn dual_walk_respects_w_lo_offset() {
        let vol = OverlayVolume::new(
            ConstVolume(200).into_volume(),
            StepVolume.into_volume(),
            OverlayColoring::default(),
        );
        let mut comp = alpha_overlay_compositor();
        // Start the walk at w=8 (x=8): already inside the opaque region.
        let color = vol.composite_color_along_normal([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], 8.0, 10.0, 1, &mut comp, 2);
        assert!((color.r() as i32 - 200).abs() <= 1, "got {:?}", color);
    }

    #[test]
    fn dual_walk_transparent_overlay_yields_black() {
        let vol = OverlayVolume::new(
            ConstVolume(200).into_volume(),
            ConstVolume(0).into_volume(),
            OverlayColoring::default(),
        );
        let mut comp = alpha_overlay_compositor();
        let color = vol.composite_color_along_normal([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], 0.0, 10.0, 1, &mut comp, 10);
        assert_eq!(color, Color32::from_gray(0));
    }

    #[test]
    fn dual_walk_empty_segment_yields_black() {
        let vol = OverlayVolume::new(
            ConstVolume(200).into_volume(),
            ConstVolume(255).into_volume(),
            OverlayColoring::default(),
        );
        let mut comp = alpha_overlay_compositor();
        let color = vol.composite_color_along_normal([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], 5.0, 5.0, 1, &mut comp, 0);
        assert_eq!(color, Color32::from_gray(0));
    }

    #[test]
    fn start_walk_runs_regular_alpha_from_overlay_onset() {
        let vol = OverlayVolume::new(
            GradientVolume.into_volume(),
            StepVolume.into_volume(),
            OverlayColoring::default(),
        );
        let mut comp = Compositor::AlphaOverlayStart(alpha_state());
        let actual = vol.composite_color_along_normal([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], 0.0, 10.0, 1, &mut comp, 10);

        // Same walk run directly on the base, starting where the overlay
        // turns significant (x=5).
        let base = GradientVolume.into_volume();
        let mut alpha = Compositor::Alpha(alpha_state());
        let expected = base.composite_color_along_normal([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], 5.0, 10.0, 1, &mut alpha, 10);
        assert_eq!(actual, expected);

        // Sanity: the onset actually mattered (full walk gives a different result).
        let mut alpha_full = Compositor::Alpha(alpha_state());
        let full = base.composite_color_along_normal([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], 0.0, 10.0, 1, &mut alpha_full, 10);
        assert_ne!(actual, full);
    }

    #[test]
    fn start_walk_falls_back_to_full_walk_when_overlay_never_fires() {
        let vol = OverlayVolume::new(
            GradientVolume.into_volume(),
            ConstVolume(0).into_volume(),
            OverlayColoring::default(),
        );
        let mut comp = Compositor::AlphaOverlayStart(alpha_state());
        let actual = vol.composite_color_along_normal([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], 0.0, 10.0, 1, &mut comp, 10);

        let base = GradientVolume.into_volume();
        let mut alpha = Compositor::Alpha(alpha_state());
        let expected = base.composite_color_along_normal([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], 0.0, 10.0, 1, &mut alpha, 10);
        assert_eq!(actual, expected);
        assert_ne!(actual, Color32::from_gray(0));
    }

    #[test]
    fn combined_walk_matches_regular_alpha_when_overlay_saturated() {
        let vol = OverlayVolume::new(
            GradientVolume.into_volume(),
            ConstVolume(255).into_volume(),
            OverlayColoring::default(),
        );
        let mut comp = Compositor::AlphaOverlayCombined(alpha_state());
        let actual = vol.composite_color_along_normal([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], 0.0, 10.0, 1, &mut comp, 10);

        let base = GradientVolume.into_volume();
        let mut alpha = Compositor::Alpha(alpha_state());
        let expected = base.composite_color_along_normal([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], 0.0, 10.0, 1, &mut alpha, 10);
        assert_eq!(actual, expected);
        assert_ne!(actual, Color32::from_gray(0));
    }

    #[test]
    fn combined_walk_zero_overlay_contributes_nothing() {
        let vol = OverlayVolume::new(
            GradientVolume.into_volume(),
            ConstVolume(0).into_volume(),
            OverlayColoring::default(),
        );
        let mut comp = Compositor::AlphaOverlayCombined(alpha_state());
        let color = vol.composite_color_along_normal([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], 0.0, 10.0, 1, &mut comp, 10);
        assert_eq!(color, Color32::from_gray(0));
    }
}
