pub mod composition;
mod empty;
mod generic;
mod grid500;
mod layers;
mod objvolume;
mod overlay;
mod ppmvolume;
mod tifxyz;
mod transform;
mod volume64x4;
mod zarr_paint;

pub use composition::{
    AlphaCompositionState, AlphaHeightMapCompositionState, CompositionState, Compositor, CompositorRef,
    MaxCompositionState, NoCompositionState,
};

use ecolor::Color32;
pub use empty::EmptyVolume;
pub use generic::AutoPaintVolume;
pub use grid500::VolumeGrid500Mapped;
pub use layers::LayersMappedVolume;
use libm::modf;
pub use objvolume::{ObjFile, ObjVolume, ProjectionKind};
pub use overlay::{BlendMode, OverlayColoring, OverlayPaintVolume, OverlayVolume};
pub use ppmvolume::PPMVolume;
use std::sync::Arc;
pub use tifxyz::{TifXyzBase, TifXyzVolume};
pub use transform::AffineTransform;
pub use volume64x4::VolumeGrid64x4Mapped;
pub use zarr_paint::{ColorScheme, FourColors, GrayScale, OmeZarrPaintVolume};

#[derive(Copy, Debug, Clone, PartialEq, Eq, Hash, serde::Deserialize, serde::Serialize)]
pub enum CompositingMode {
    None,
    Max,
    Alpha,
    AlphaHeightMap,
    /// Alpha walk where the overlay volume supplies per-sample opacity and the
    /// base volume supplies the value. Only `OverlayVolume` interprets this
    /// specially (see its `composite_color_along_normal`); for plain volumes it
    /// behaves exactly like `Alpha`.
    AlphaOverlay,
}
impl CompositingMode {
    pub fn label(&self) -> &str {
        match self {
            CompositingMode::None => "None",
            CompositingMode::Max => "Max",
            CompositingMode::Alpha => "Alpha",
            CompositingMode::AlphaHeightMap => "Alpha Height Map",
            CompositingMode::AlphaOverlay => "Alpha (overlay opacity)",
        }
    }
    pub const VALUES: [CompositingMode; 5] = [
        CompositingMode::None,
        CompositingMode::Max,
        CompositingMode::Alpha,
        CompositingMode::AlphaHeightMap,
        CompositingMode::AlphaOverlay,
    ];
}
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Deserialize, serde::Serialize)]
pub struct CompositingSettings {
    pub mode: CompositingMode,
    pub layers_in_front: u8,
    pub layers_behind: u8,
    pub alpha_min: u8,
    pub alpha_max: u8,
    pub alpha_threshold: u16,
    pub opacity: u16,
    pub reverse_direction: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Deserialize, serde::Serialize)]
#[serde(default)]
pub struct DrawingConfig {
    pub enable_filters: bool,
    pub threshold_min: u8,
    pub threshold_max: u8,
    pub quant: u8,
    pub mask_shift: u8,
    pub trilinear_interpolation: bool,
    pub draw_xyz_outlines: bool,
    pub show_segment_outlines: bool,
    pub draw_outline_vertices: bool,
    pub compositing: CompositingSettings,
    /// Debug: tint each painted chunk by its cache state.
    pub debug_chunk_overlay: bool,
}
impl DrawingConfig {
    pub fn filters_active(&self) -> bool {
        self.enable_filters
            && (self.threshold_min > 0 || self.threshold_max > 0 || self.quant < 8 || self.mask_shift > 0)
    }
    pub fn bit_mask(&self) -> u8 {
        (match self.quant {
            8 => 0xff,
            7 => 0xfe,
            6 => 0xfc,
            5 => 0xf8,
            4 => 0xf0,
            3 => 0xe0,
            2 => 0xc0,
            1 => 0x80,
            _ => 0xff,
        }) >> self.mask_shift
    }
    pub fn filter(&self, value: u8) -> u8 {
        if self.filters_active() {
            let pluscon = ((value as i32 - self.threshold_min as i32).max(0) * 255
                / (255 - (self.threshold_min + self.threshold_max) as i32))
                .min(255) as u8;

            (((pluscon & self.bit_mask()) as f32) / (self.bit_mask() as f32) * 255.0) as u8
        } else {
            value
        }
    }
}
impl Default for DrawingConfig {
    fn default() -> Self {
        Self {
            enable_filters: false,
            threshold_min: 0,
            threshold_max: 0,
            quant: 0xff,
            mask_shift: 0,
            trilinear_interpolation: false,
            draw_xyz_outlines: false,
            show_segment_outlines: true,
            draw_outline_vertices: false,
            compositing: CompositingSettings {
                mode: CompositingMode::None,
                layers_in_front: 6,
                layers_behind: 6,
                alpha_min: (0.3 * 255.0) as u8,
                alpha_max: (0.7 * 255.0) as u8,
                alpha_threshold: 9500,
                opacity: 1,
                reverse_direction: false,
            },
            debug_chunk_overlay: false,
        }
    }
}

pub trait VoxelVolume {
    fn reset_for_painting(&self) {}

    /// Pre-touch every chunk inside the axis-aligned voxel-coord box
    /// `[min, max]` at the LOD selected by `downsampling`. Backends that
    /// stream chunks asynchronously (the unified cache) use this to
    /// kick dispatch + upscale-from-parent for a whole triangle's worth
    /// of samples before the per-voxel composite loop runs. Default
    /// impl: no-op (in-memory backends already have everything).
    fn touch_aabb(&self, _min: [f64; 3], _max: [f64; 3], _downsampling: i32) {}

    fn get(&self, xyz: [f64; 3], downsampling: i32) -> u8;

    fn get_interpolated(&self, xyz: [f64; 3], downsampling: i32) -> u8 {
        self.get_interpolated_slow(xyz, downsampling)
    }

    fn get_color(&self, xyz: [f64; 3], downsampling: i32) -> Color32 {
        Color32::from_gray(self.get(xyz, downsampling))
    }

    fn get_color_interpolated(&self, xyz: [f64; 3], downsampling: i32) -> Color32 {
        Color32::from_gray(self.get_interpolated(xyz, downsampling))
    }

    fn get_interpolated_slow(&self, xyz: [f64; 3], downsampling: i32) -> u8 {
        let (dx, x0) = modf(xyz[0]);
        let x1 = x0 + 1.0;
        let (dy, y0) = modf(xyz[1]);
        let y1 = y0 + 1.0;
        let (dz, z0) = modf(xyz[2]);
        let z1 = z0 + 1.0;

        let c00 =
            self.get([x0, y0, z0], downsampling) as f64 * (1.0 - dx) + self.get([x1, y0, z0], downsampling) as f64 * dx;
        let c10 =
            self.get([x0, y1, z0], downsampling) as f64 * (1.0 - dx) + self.get([x1, y1, z0], downsampling) as f64 * dx;
        let c01 =
            self.get([x0, y0, z1], downsampling) as f64 * (1.0 - dx) + self.get([x1, y0, z1], downsampling) as f64 * dx;
        let c11 =
            self.get([x0, y1, z1], downsampling) as f64 * (1.0 - dx) + self.get([x1, y1, z1], downsampling) as f64 * dx;

        let c0 = c00 * (1.0 - dy) + c10 * dy;
        let c1 = c01 * (1.0 - dy) + c11 * dy;

        let c = c0 * (1.0 - dz) + c1 * dz;

        c as u8
    }

    /// Walk integer-step trilinear samples along `base + w * dir` for
    /// `w in w_lo, w_lo+1, …, w_hi-1`, feeding each sample to the typed
    /// compositor. `update` returning `false` stops the walk early
    /// (lets alpha compositing bail once it hits saturation).
    ///
    /// `CompositorRef` is an enum over the concrete state types: the
    /// cache override matches once at the top of the call and dispatches
    /// to a monomorphized inner loop per arm, so the per-sample update
    /// folds into the trilerp body without any virtual call.
    ///
    /// The default impl is the same per-sample loop the call site used to
    /// inline. Backends that resolve to a chunked cache override this to
    /// amortize chunk lookups across the whole walk.
    fn composite_along_normal(
        &self,
        base: [f64; 3],
        dir: [f64; 3],
        w_lo: f64,
        w_hi: f64,
        downsampling: i32,
        compositor: &mut CompositorRef<'_>,
    ) {
        let n = (w_hi - w_lo) as i32;
        for k in 0..n {
            let w = w_lo + k as f64;
            let p = [base[0] + w * dir[0], base[1] + w * dir[1], base[2] + w * dir[2]];
            let v = self.get_interpolated(p, downsampling);
            if !compositor.update(v) {
                return;
            }
        }
    }

    /// Composite along a ray and return a colored result. Used by paint
    /// loops that need a colored output (overlays). The default impl runs
    /// the regular `composite_along_normal` with the caller's compositor
    /// and wraps the result in `Color32::from_gray`, so backends that
    /// override the gray fast path also get the colored fast path for
    /// free. `OverlayVolume` overrides this to run two parallel fast-path
    /// walks (base + overlay) and blend the results, keeping the cache
    /// fast path on the base while still painting the overlay tint.
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
        self.composite_along_normal(
            base,
            dir,
            w_lo,
            w_hi,
            downsampling,
            &mut compositor.as_ref_mut(),
        );
        Color32::from_gray(compositor.result(num_layers))
    }

    /// Gather `out.len()` trilinear samples along `base + k*dir` for
    /// `k in 0..out.len()` into `out`. This is the "no compositor" form of
    /// the ray walk: callers that need the raw per-sample values (e.g. to
    /// combine two volumes — value from one, opacity from another) use this
    /// to keep the cache's amortized chunk lookup instead of calling
    /// `get_interpolated` per sample. Default impl is the per-sample slow
    /// loop; chunked backends override it.
    fn gather_along_normal(&self, base: [f64; 3], dir: [f64; 3], downsampling: i32, out: &mut [u8]) {
        for (k, slot) in out.iter_mut().enumerate() {
            let w = k as f64;
            let p = [base[0] + w * dir[0], base[1] + w * dir[1], base[2] + w * dir[2]];
            *slot = self.get_interpolated(p, downsampling);
        }
    }
}

pub struct Image {
    pub width: usize,
    pub height: usize,
    pub data: Vec<Color32>,
}
impl Image {
    pub fn new(width: usize, height: usize) -> Self {
        Self::new_from_color(width, height, Color32::BLACK)
    }
    pub fn new_from_color(width: usize, height: usize, color: Color32) -> Self {
        Self {
            width,
            height,
            data: vec![color; width * height],
        }
    }

    pub fn set(&mut self, x: usize, y: usize, value: Color32) {
        self.data[y * self.width + x] = value;
    }
    pub fn set_rgb(&mut self, x: usize, y: usize, r: u8, g: u8, b: u8) {
        self.set(x, y, Color32::from_rgb(r, g, b));
    }
    pub fn set_gray(&mut self, x: usize, y: usize, value: u8) {
        self.set(x, y, Color32::from_gray(value));
    }
    pub fn blend(&mut self, x: usize, y: usize, value: Color32, alpha: f32) {
        let pos = &mut self.data[y * self.width + x];
        let old = *pos;
        *pos = Color32::from_rgba_unmultiplied(
            (old.r() as f32 * (1.0 - alpha) + value.r() as f32 * alpha) as u8,
            (old.g() as f32 * (1.0 - alpha) + value.g() as f32 * alpha) as u8,
            (old.b() as f32 * (1.0 - alpha) + value.b() as f32 * alpha) as u8,
            (old.a() as f32 * (1.0 - alpha) + value.a() as f32 * alpha) as u8,
        );
    }
}

pub type VolumeCons = Box<dyn (FnOnce() -> Volume) + Send + Sync>;

pub trait PaintVolume {
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
    );

    fn shared(&self) -> VolumeCons;
}

pub trait VoxelPaintVolume: PaintVolume + VoxelVolume {
    fn into_volume(self) -> Volume
    where
        Self: Sized + 'static,
    {
        Volume::new(self)
    }
}
impl<T: PaintVolume + VoxelVolume> VoxelPaintVolume for T {}

pub trait SurfaceVolume: PaintVolume + VoxelVolume {
    fn paint_plane_intersection(
        &self,
        xyz: [i32; 3],
        u_coord: usize,
        v_coord: usize,
        plane_coord: usize,
        width: usize,
        height: usize,
        sfactor: u8,
        paint_zoom: u8,
        highlight_uv_section: Option<[i32; 3]>,
        config: &DrawingConfig,
        buffer: &mut Image,
    );
}

#[derive(Clone)]
pub struct Volume {
    pub volume: Arc<dyn VoxelPaintVolume>,
}
impl Volume {
    pub fn new(volume: impl VoxelPaintVolume + 'static) -> Self {
        Self {
            volume: Arc::new(volume),
        }
    }
    pub fn from_ref(volume: Arc<dyn VoxelPaintVolume>) -> Self {
        Self { volume }
    }
}
impl PaintVolume for Volume {
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
        self.volume.paint(
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
        self.volume.shared()
    }
}
impl VoxelVolume for Volume {
    fn get(&self, xyz: [f64; 3], downsampling: i32) -> u8 {
        self.volume.get(xyz, downsampling)
    }
    fn get_interpolated(&self, xyz: [f64; 3], downsampling: i32) -> u8 {
        self.volume.get_interpolated(xyz, downsampling)
    }
    fn get_color(&self, xyz: [f64; 3], downsampling: i32) -> Color32 {
        self.volume.get_color(xyz, downsampling)
    }
    fn get_color_interpolated(&self, xyz: [f64; 3], downsampling: i32) -> Color32 {
        self.volume.get_color_interpolated(xyz, downsampling)
    }
    fn reset_for_painting(&self) {
        self.volume.reset_for_painting();
    }
    fn touch_aabb(&self, min: [f64; 3], max: [f64; 3], downsampling: i32) {
        self.volume.touch_aabb(min, max, downsampling);
    }
    fn composite_along_normal(
        &self,
        base: [f64; 3],
        dir: [f64; 3],
        w_lo: f64,
        w_hi: f64,
        downsampling: i32,
        compositor: &mut CompositorRef<'_>,
    ) {
        self.volume
            .composite_along_normal(base, dir, w_lo, w_hi, downsampling, compositor);
    }
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
        self.volume.composite_color_along_normal(
            base,
            dir,
            w_lo,
            w_hi,
            downsampling,
            compositor,
            num_layers,
        )
    }
    fn gather_along_normal(&self, base: [f64; 3], dir: [f64; 3], downsampling: i32, out: &mut [u8]) {
        self.volume.gather_along_normal(base, dir, downsampling, out);
    }
}
