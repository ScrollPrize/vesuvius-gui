use crate::volume::{DrawingConfig, Image, PaintVolume, VolumeCons, VoxelPaintVolume, VoxelVolume};
use ecolor::Color32;
use std::marker::PhantomData;
use vesuvius_zarr::{OmeZarrContext, ZarrContext};

pub trait ColorScheme {
    fn get_color(value: u8) -> Color32;
}

pub struct FourColors {}
impl ColorScheme for FourColors {
    fn get_color(value: u8) -> Color32 {
        match value {
            1 => Color32::RED,
            2 => Color32::GREEN,
            3 => Color32::YELLOW,
            _ => Color32::BLUE,
        }
    }
}

pub struct GrayScale {}
impl ColorScheme for GrayScale {
    fn get_color(value: u8) -> Color32 {
        Color32::from_gray(value)
    }
}

impl PaintVolume for ZarrContext<3> {
    fn paint(
        &self,
        xyz: [i32; 3],
        u_coord: usize,
        v_coord: usize,
        plane_coord: usize,
        width: usize,
        height: usize,
        _sfactor: u8,
        paint_zoom: u8,
        _config: &DrawingConfig,
        buffer: &mut Image,
    ) {
        let _sfactor = 1;
        if !self.cache_missing() {
            self.purge_missing_cache();
        }

        let fi32 = _sfactor as f64;

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

                let v = self.get([z as usize, y as usize, x as usize]).unwrap_or(0);
                if v != 0 {
                    buffer.set_gray(im_u, im_v, v);
                }
            }
        }
    }

    fn shared(&self) -> VolumeCons {
        let cons = self.shareable();
        Box::new(move || cons().into_volume())
    }
}

impl VoxelVolume for ZarrContext<3> {
    fn get(&self, xyz: [f64; 3], downsampling: i32) -> u8 {
        ZarrContext::get(
            self,
            [
                (xyz[2] * downsampling as f64) as usize,
                (xyz[1] * downsampling as f64) as usize,
                (xyz[0] * downsampling as f64) as usize,
            ],
        )
        .unwrap_or(0)
    }
    fn get_interpolated(&self, xyz: [f64; 3], downsampling: i32) -> u8 {
        ZarrContext::get_interpolated(
            self,
            [
                xyz[2] * downsampling as f64,
                xyz[1] * downsampling as f64,
                xyz[0] * downsampling as f64,
            ],
        )
        .unwrap_or(0)
    }
    fn reset_for_painting(&self) {
        self.purge_missing_cache();
    }
}

pub struct OmeZarrPaintVolume<C: ColorScheme> {
    inner: OmeZarrContext,
    _phantom: PhantomData<C>,
}

impl<C: ColorScheme> OmeZarrPaintVolume<C> {
    pub fn new(inner: OmeZarrContext) -> Self {
        Self {
            inner,
            _phantom: PhantomData,
        }
    }
}

impl<C: ColorScheme + 'static + Send + Sync> PaintVolume for OmeZarrPaintVolume<C> {
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
        if !self.inner.cache_missing {
            self.inner.zarr_contexts.iter().for_each(|ctx| {
                ctx.purge_missing_cache();
            });
        }

        let scale = sfactor.trailing_zeros() as u8;

        for im_u in 0..width {
            for im_v in 0..height {
                let im_rel_u = (im_u as i32 - width as i32 / 2) * paint_zoom as i32;
                let im_rel_v = (im_v as i32 - height as i32 / 2) * paint_zoom as i32;

                let mut uvw: [f64; 3] = [0.; 3];
                uvw[u_coord] = (xyz[u_coord] + im_rel_u) as f64;
                uvw[v_coord] = (xyz[v_coord] + im_rel_v) as f64;
                uvw[plane_coord] = (xyz[plane_coord]) as f64;

                let [x, y, z] = uvw;

                if x < 0.0 || y < 0.0 || z < 0.0 {
                    continue;
                }

                let v = self.inner.get([z as usize, y as usize, x as usize], scale);
                if v != 0 {
                    let v = config.filter(v);
                    buffer.set(im_u, im_v, C::get_color(v));
                }
            }
        }
    }

    fn shared(&self) -> VolumeCons {
        let ome_zarr = self.inner.ome_zarr.clone();
        let cache_missing = self.inner.cache_missing;
        let zarr_contexts = self
            .inner
            .zarr_contexts
            .iter()
            .map(|ctx| ctx.shareable())
            .collect::<Vec<_>>();

        Box::new(move || {
            let inner = OmeZarrContext {
                ome_zarr: ome_zarr.clone(),
                cache_missing,
                zarr_contexts: zarr_contexts.into_iter().map(|ctx| ctx()).collect(),
            };
            OmeZarrPaintVolume::<C>::new(inner).into_volume()
        })
    }
}

impl<C: ColorScheme> VoxelVolume for OmeZarrPaintVolume<C> {
    fn get(&self, xyz: [f64; 3], downsampling: i32) -> u8 {
        let scale = downsampling.trailing_zeros() as u8;
        self.inner.get(
            [
                (xyz[2] * downsampling as f64) as usize,
                (xyz[1] * downsampling as f64) as usize,
                (xyz[0] * downsampling as f64) as usize,
            ],
            scale,
        )
    }
    fn get_interpolated(&self, xyz: [f64; 3], downsampling: i32) -> u8 {
        let scale = downsampling.trailing_zeros() as u8;
        self.inner.get_interpolated(
            [
                xyz[2] * downsampling as f64,
                xyz[1] * downsampling as f64,
                xyz[0] * downsampling as f64,
            ],
            scale,
        )
    }
    fn reset_for_painting(&self) {
        self.inner.zarr_contexts.iter().for_each(|ctx| {
            ctx.reset_for_painting();
        });
    }
}
