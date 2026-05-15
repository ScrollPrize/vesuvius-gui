use super::zarr_paint::{ColorScheme, FourColors};
use super::{DrawingConfig, Image, PaintVolume, Volume, VolumeCons, VoxelPaintVolume, VoxelVolume};
use ecolor::Color32;

#[derive(Debug, Clone, Copy, serde::Deserialize, serde::Serialize)]
pub enum OverlayColoring {
    /// 1→red, 2→green, 3→yellow, 4+→blue; 0 → no paint.
    FourColors { alpha: f32 },
    /// value 255 → `color`, anything else → no paint. Color stored as [r, g, b]
    /// so the enum can derive serde (ecolor's Color32 lacks serde by default).
    Boolean { color: [u8; 3], alpha: f32 },
    /// 0 → no paint; otherwise color = HSV(hue_deg, 1, value/255).
    Hue { hue_deg: f32, alpha: f32 },
}

impl OverlayColoring {
    pub fn paint(&self, value: u8) -> Option<(Color32, f32)> {
        match *self {
            OverlayColoring::FourColors { alpha } => {
                if value == 0 {
                    None
                } else {
                    Some((FourColors::get_color(value), alpha))
                }
            }
            OverlayColoring::Boolean { color, alpha } => {
                if value == 255 {
                    Some((Color32::from_rgb(color[0], color[1], color[2]), alpha))
                } else {
                    None
                }
            }
            OverlayColoring::Hue { hue_deg, alpha } => {
                if value == 0 {
                    None
                } else {
                    Some((hsv_to_color32(hue_deg, 1.0, value as f32 / 255.0), alpha))
                }
            }
        }
    }
}

impl Default for OverlayColoring {
    fn default() -> Self {
        OverlayColoring::FourColors { alpha: 0.4 }
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

fn blend_color32(base: Color32, value: Color32, alpha: f32) -> Color32 {
    Color32::from_rgba_unmultiplied(
        (base.r() as f32 * (1.0 - alpha) + value.r() as f32 * alpha) as u8,
        (base.g() as f32 * (1.0 - alpha) + value.g() as f32 * alpha) as u8,
        (base.b() as f32 * (1.0 - alpha) + value.b() as f32 * alpha) as u8,
        (base.a() as f32 * (1.0 - alpha) + value.a() as f32 * alpha) as u8,
    )
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
        self.inner.reset_for_painting();

        let fi32 = sfactor as f64;

        for im_u in 0..width {
            for im_v in 0..height {
                let im_rel_u = (im_u as i32 - width as i32 / 2) * paint_zoom as i32;
                let im_rel_v = (im_v as i32 - height as i32 / 2) * paint_zoom as i32;

                let mut uvw: [f64; 3] = [0.; 3];
                uvw[u_coord] = (xyz[u_coord] + im_rel_u) as f64 / fi32;
                uvw[v_coord] = (xyz[v_coord] + im_rel_v) as f64 / fi32;
                uvw[plane_coord] = (xyz[plane_coord]) as f64 / fi32;

                let [x, y, z] = uvw;

                if x < 0.0 || y < 0.0 || z < 0.0 {
                    continue;
                }

                let v = self.inner.get([x, y, z], sfactor as i32);
                if let Some((color, alpha)) = self.coloring.paint(v) {
                    buffer.blend(im_u, im_v, color, alpha);
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
        Self { base, overlay, coloring }
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
            Some((c, a)) => blend_color32(base_color, c, a),
            None => base_color,
        }
    }
    fn get_color_interpolated(&self, xyz: [f64; 3], downsampling: i32) -> Color32 {
        let base_color = self.base.get_color_interpolated(xyz, downsampling);
        let lab = self.overlay.get_interpolated(xyz, downsampling);
        match self.coloring.paint(lab) {
            Some((c, a)) => blend_color32(base_color, c, a),
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
            xyz, u_coord, v_coord, plane_coord, width, height, sfactor, paint_zoom, config, buffer,
        );
        OverlayPaintVolume::new(self.overlay.clone(), self.coloring).paint(
            xyz, u_coord, v_coord, plane_coord, width, height, sfactor, paint_zoom, config, buffer,
        );
    }

    fn shared(&self) -> VolumeCons {
        let base_cons = self.base.shared();
        let overlay_cons = self.overlay.shared();
        let coloring = self.coloring;
        Box::new(move || OverlayVolume::new(base_cons(), overlay_cons(), coloring).into_volume())
    }
}
