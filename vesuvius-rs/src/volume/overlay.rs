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
