use anyhow::{anyhow, Result};
use clap::Parser;
use futures::{stream, StreamExt};
use image::Luma;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use std::cell::RefCell;
use std::collections::BTreeSet;
use std::ops::RangeInclusive;
use std::path::PathBuf;
use std::rc::Rc;
use std::sync::Arc;
use std::time::{Duration, Instant};
use vesuvius_rs::cache::backfillers::ome_zarr::OmeZarrBackfiller;
use vesuvius_rs::cache::backfillers::synthesized_lod::SynthesizedLodBackfiller;
use vesuvius_rs::cache::epoch::CAP_ENV_VAR;
use vesuvius_rs::cache::{ChunkBackfiller, ChunkCache, ChunkKey, ChunkState, UnifiedCache, UnifiedVolume};
use vesuvius_rs::model::{NewVolumeReference, VolumeLocation};
use vesuvius_rs::volume::{
    AffineTransform, CompositingMode, CompositingSettings, DrawingConfig, Image, ObjFile, ObjVolume, PaintVolume,
    ProjectionKind, Volume, VolumeCons, VoxelPaintVolume, VoxelVolume,
};
use vesuvius_zarr::{base_cache_dir, unified_volume_key, OmeZarrContext};

/// Wall-clock budget for making one tile's chunks resident before we give up
/// and render with whatever is available. Chunk fetches are normally far
/// faster; this only guards against a permanently failing source so the run
/// can't hang forever.
const TILE_ENSURE_TIMEOUT: Duration = Duration::from_secs(300);

#[derive(Clone, Debug)]
pub struct Crop {
    pub top: usize,
    pub left: usize,
    pub width: usize,
    pub height: usize,
}
#[derive(Clone)]
struct CropParser;
impl clap::builder::TypedValueParser for CropParser {
    type Value = Crop;

    fn parse_ref(
        &self,
        _cmd: &clap::Command,
        _arg: Option<&clap::Arg>,
        value: &std::ffi::OsStr,
    ) -> std::result::Result<Self::Value, clap::Error> {
        // parse a value like 0+0-0x0 with a regexp

        let re = regex::Regex::new(r"(\d+)\+(\d+)-(\d+)x(\d+)").unwrap();
        let captures = re.captures(value.to_str().unwrap()).ok_or(clap::Error::raw(
            clap::error::ErrorKind::ValueValidation,
            "--crop argument could not be parsed. Use '--crop <left>+<top>-<width>x<height>', e.g. '--crop 1000+1000-500x500'.",
        ))?;
        let left = captures.get(1).unwrap().as_str().parse().unwrap();
        let top = captures.get(2).unwrap().as_str().parse().unwrap();
        let width = captures.get(3).unwrap().as_str().parse().unwrap();
        let height = captures.get(4).unwrap().as_str().parse().unwrap();
        Ok(Crop {
            top,
            left,
            width,
            height,
        })
    }
}

/// Alpha / compositing options. When `--enable-alpha` is set the renderer
/// composites along the surface normal across the configured layer window
/// instead of sampling a single plane per layer.
#[derive(Parser, Debug, Clone)]
pub struct AlphaArgs {
    /// Enable alpha compositing along the surface normal
    #[clap(long, default_value_t = false)]
    enable_alpha: bool,

    /// Number of layers to composite in front of the surface (default 6)
    #[clap(long)]
    composite_layers_in_front: Option<u8>,

    /// Number of layers to composite behind the surface (default 6)
    #[clap(long)]
    composite_layers_behind: Option<u8>,

    /// Alpha ramp lower bound, 0-255 (default 76)
    #[clap(long)]
    alpha_min: Option<u8>,

    /// Alpha ramp upper bound, 0-255 (default 178)
    #[clap(long)]
    alpha_max: Option<u8>,

    /// Alpha threshold, 0-10000 (default 9500)
    #[clap(long)]
    alpha_threshold: Option<u16>,

    /// Opacity multiplier (default 1)
    #[clap(long)]
    opacity: Option<u16>,

    /// Reverse the compositing direction along the normal
    #[clap(long, default_value_t = false)]
    composite_reverse_direction: bool,
}
impl AlphaArgs {
    fn to_settings(&self) -> CompositingSettings {
        let mut settings = DrawingConfig::default().compositing;
        settings.mode = if self.enable_alpha {
            CompositingMode::Alpha
        } else {
            CompositingMode::None
        };
        if let Some(v) = self.composite_layers_in_front {
            settings.layers_in_front = v;
        }
        if let Some(v) = self.composite_layers_behind {
            settings.layers_behind = v;
        }
        if let Some(v) = self.alpha_min {
            settings.alpha_min = v;
        }
        if let Some(v) = self.alpha_max {
            settings.alpha_max = v;
        }
        if let Some(v) = self.alpha_threshold {
            settings.alpha_threshold = v;
        }
        if let Some(v) = self.opacity {
            settings.opacity = v;
        }
        settings.reverse_direction = self.composite_reverse_direction;
        settings
    }
}

/// Vesuvius Renderer, a tool to render segments from obj files
#[derive(Parser, Debug)]
#[command(about, long_about = None)]
pub struct Args {
    /// Provide segment file to render
    #[clap(long)]
    obj: String,

    /// Width of the segment file when browsing obj files
    #[clap(long)]
    width: u32,
    /// Height of the segment file when browsing obj files
    #[clap(long)]
    height: u32,

    /// Transform to apply to the obj file (to map between different scans). You can either supply a filename to a transform json file
    /// (as defined in https://github.com/ScrollPrize/villa/blob/main/foundation/volume-registration/transform_schema.json) or supply
    /// a 4x3 affine transformation matrix as a json array string directly
    #[clap(long)]
    transform: Option<String>,

    /// Use orthographic projection along the Y axis (top-down view) when loading obj files (discarding existing texture coordinates).
    #[clap(long, default_value_t = false)]
    ortho_xz: bool,

    /// Invert the transform before applying it
    #[clap(long)]
    invert_transform: bool,

    /// The target directory to save the rendered images
    #[clap(long)]
    target_dir: String,

    /// Output layer id that corresponds to the segment surface (default 32)
    #[clap(long)]
    middle_layer: Option<u8>,

    /// Minimum layer id to render (default 25)
    #[clap(long)]
    min_layer: Option<u8>,

    /// Maximum layer id to render (default 41)
    #[clap(long)]
    max_layer: Option<u8>,

    /// Crop a specific region from the segment. The format is <left>+<top>-<width>x<height>.
    #[clap(long, value_parser = CropParser)]
    crop: Option<Crop>,

    /// File extension / image format to use for layers (default png)
    #[clap(long)]
    target_format: Option<String>,

    /// The ome-zarr volume to render against, given as an http(s) URL or a local path.
    #[clap(short, long)]
    volume: Option<String>,

    /// Override the data directory. By default, a directory in the user's cache is used
    #[clap(short, long)]
    data_directory: Option<String>,

    /// The tile size to split a segment into (for ergonomic reasons) (default 1024)
    #[clap(long)]
    tile_size: Option<u32>,

    /// Number of tiles to render in parallel (CPU-bound stage). Defaults to
    /// the number of CPU threads.
    #[clap(long)]
    render_concurrency: Option<usize>,

    /// How many tiles to fetch/ensure ahead of rendering. The ensure stage
    /// runs `render_concurrency + prefetch_depth` tiles concurrently, so the
    /// next tiles' chunks download while the current ones render. Together
    /// these bound the cache working set (~(render_concurrency +
    /// prefetch_depth) tiles); lower them if the purge log shows repeated
    /// evict/refetch cycles (default 4).
    #[clap(long)]
    prefetch_depth: Option<usize>,

    /// Override the unified cache size cap, in GB (sets VESUVIUS_CACHE_CAP_GB).
    /// Raise this for large renders so the working set fits without purging.
    #[clap(long)]
    cache_cap_gb: Option<u64>,

    /// CPU-bound blocking threads for the tokio runtime (default:
    /// render_concurrency + prefetch_depth + 2). Only the collect and render
    /// stages use blocking threads; the ensure stage waits asynchronously.
    #[clap(long)]
    worker_threads: Option<usize>,

    #[clap(flatten)]
    alpha: AlphaArgs,
}

fn main() -> Result<()> {
    env_logger::init();
    let args = Args::parse();
    // The blocking pool runs the collect + render stages; the ensure stage's
    // wait is async and holds no blocking thread. Give the pool room for both
    // the renders and the prefetch tiles' collects so prefetch isn't starved.
    let render_conc = args.render_concurrency.unwrap_or(num_cpus::get());
    let prefetch = args.prefetch_depth.unwrap_or(4);
    let threads = args.worker_threads.unwrap_or(render_conc + prefetch + 2);

    // The cache cap is read once (cached) the first time a cache is built, so
    // it has to be set before any cache construction happens.
    if let Some(gb) = args.cache_cap_gb {
        std::env::set_var(CAP_ENV_VAR, gb.to_string());
    }

    let result = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .max_blocking_threads(threads)
        .build()
        .unwrap()
        .block_on(main_run(args));

    result
}

async fn main_run(args: Args) -> Result<()> {
    let multi = MultiProgress::new();
    monitor_runtime_stats(&multi).await;

    let params = (&args).into();
    let rendering = Rendering::new(params, &args)?;
    rendering.run(&multi).await?;
    Ok(())
}

#[derive(Clone)]
struct RenderParams {
    width: usize,
    height: usize,
    projection: ProjectionKind,
    tile_size: usize,
    w_range: RangeInclusive<usize>,
    crop: Option<Crop>,
    mid_layer: usize,
    target_dir: String,
    target_format: String,
    compositing: CompositingSettings,
    render_concurrency: usize,
    prefetch_depth: usize,
}
impl RenderParams {
    fn render_left(&self) -> usize {
        self.crop.as_ref().map(|c| c.left).unwrap_or(0)
    }
    fn render_top(&self) -> usize {
        self.crop.as_ref().map(|c| c.top).unwrap_or(0)
    }
    fn render_width(&self) -> usize {
        self.crop.as_ref().map(|c| c.width).unwrap_or(self.width)
    }
    fn render_height(&self) -> usize {
        self.crop.as_ref().map(|c| c.height).unwrap_or(self.height)
    }
    /// The drawing config shared by the chunk-collection pass and the render
    /// pass — they MUST agree so the chunks we make resident are exactly the
    /// ones the render samples.
    fn drawing_config(&self) -> DrawingConfig {
        let mut config = DrawingConfig::default();
        config.trilinear_interpolation = true;
        config.compositing = self.compositing.clone();
        config
    }
}
impl From<&Args> for RenderParams {
    fn from(args: &Args) -> Self {
        Self {
            width: args.width as usize,
            height: args.height as usize,
            projection: if args.ortho_xz {
                ProjectionKind::OrthographicXZ
            } else {
                ProjectionKind::None
            },
            tile_size: args.tile_size.unwrap_or(1024) as usize,
            w_range: args.min_layer.unwrap_or(25) as usize..=args.max_layer.unwrap_or(41) as usize,
            crop: args.crop.clone(),
            mid_layer: args.middle_layer.unwrap_or(32) as usize,
            target_dir: args.target_dir.clone(),
            target_format: args.target_format.clone().unwrap_or("png".to_string()),
            compositing: args.alpha.to_settings(),
            render_concurrency: args.render_concurrency.unwrap_or(num_cpus::get()).max(1),
            prefetch_depth: args.prefetch_depth.unwrap_or(4),
        }
    }
}

#[derive(Debug, Copy, Clone)]
struct UVTile {
    u: usize,
    v: usize,
    w: usize,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct VolumeChunk {
    x: usize,
    y: usize,
    z: usize,
}
impl From<(usize, usize, usize)> for VolumeChunk {
    fn from((x, y, z): (usize, usize, usize)) -> Self {
        VolumeChunk { x, y, z }
    }
}
impl VolumeChunk {
    fn key(&self) -> ChunkKey {
        ChunkKey::new(0, self.x as u32, self.y as u32, self.z as u32)
    }
}

#[derive(Clone)]
struct Rendering {
    params: RenderParams,
    obj: Arc<ObjFile>,
    cache: ChunkCache,
}

impl Rendering {
    fn new(params: RenderParams, args: &Args) -> Result<Self> {
        let obj = Arc::new(ObjVolume::load_obj(&args.obj, &resolve_transform(args)?, params.projection));

        let cache_root = args
            .data_directory
            .clone()
            .map(PathBuf::from)
            .unwrap_or_else(base_cache_dir);

        let volume_arg = args
            .volume
            .clone()
            .ok_or_else(|| anyhow!("--volume <ome-zarr url or local path> is required"))?;
        let cache = build_cache(&volume_arg, cache_root)?;

        Ok(Self { params, obj, cache })
    }

    async fn run(&self, multi: &MultiProgress) -> Result<()> {
        std::fs::create_dir_all(&self.params.target_dir)?;

        let count_style = ProgressStyle::with_template(
            "{spinner} {msg:25} {bar:80.cyan/blue} [{elapsed_precise}] ({eta:>4}) {pos}/{len}",
        )
        .unwrap()
        .tick_chars("→↘↓↙←↖↑↗");

        let tiles = self.uv_tiles();
        let tiles_per_layer = tiles.len() as u64 / self.params.w_range.clone().count() as u64;

        let ensure_bar = ProgressBar::new(tiles.len() as u64)
            .with_style(count_style.clone().tick_chars("⠁⠂⠄⡀⢀⠠⠐⠈"))
            .with_message("Ensuring chunks");
        multi.add(ensure_bar.clone());
        ensure_bar.tick();

        let render_bar = ProgressBar::new(tiles.len() as u64)
            .with_style(count_style.clone().tick_chars("▪▫▨▧▦▩"))
            .with_message("Rendering tiles");
        multi.add(render_bar.clone());
        render_bar.tick();

        let layers_bar = ProgressBar::new(self.params.w_range.clone().count() as u64)
            .with_style(count_style.tick_chars("⌷⌸⌹⌺"))
            .with_message("Saving layers");
        multi.add(layers_bar.clone());
        layers_bar.tick();

        // Two bounded stages. The ensure stage runs ahead of rendering by
        // `prefetch_depth` tiles so the next tiles' chunks download (in the
        // cache's pool) while the current ones render — but only that far
        // ahead, so the resident working set stays small enough that the
        // background purge doesn't evict a tile's chunks before we read them.
        let render_conc = self.params.render_concurrency.max(1);
        let ensure_conc = (render_conc + self.params.prefetch_depth).max(1);

        stream::iter(tiles)
            .map(|tile| {
                let self_clone = self.clone();
                let ensure_bar = ensure_bar.clone();
                async move {
                    // Collect is CPU work (a paint pass); the ensure wait is
                    // async so it doesn't tie up a blocking thread.
                    let chunks = {
                        let s = self_clone.clone();
                        tokio::task::spawn_blocking(move || s.chunks_for(&tile)).await.unwrap()
                    };
                    self_clone.ensure_resident(&chunks).await;
                    ensure_bar.inc(1);
                    tile
                }
            })
            .buffered(ensure_conc)
            .map(|tile| {
                let self_clone = self.clone();
                let render_bar = render_bar.clone();
                async move {
                    let tile_image = tokio::task::spawn_blocking(move || self_clone.render_tile(&tile))
                        .await
                        .unwrap();
                    render_bar.inc(1);
                    (tile, tile_image)
                }
            })
            .buffered(render_conc) // ordered so each `chunks()` group is exactly one layer
            .chunks(tiles_per_layer as usize)
            .map(|tiles| {
                let self_clone = self.clone();
                let layers_bar = layers_bar.clone();
                async move {
                    let res = tokio::task::spawn_blocking(move || self_clone.render_layer_from_tiles(tiles))
                        .await
                        .unwrap();
                    layers_bar.inc(1);
                    res
                }
            })
            .buffer_unordered(render_conc)
            .collect::<Vec<_>>()
            .await;

        // Make sure the chunks we fetched are durably recorded for the next run.
        self.cache.flush();

        Ok(())
    }

    fn uv_tiles(&self) -> Vec<UVTile> {
        let top = self.params.crop.as_ref().map(|c| c.top).unwrap_or(0);
        let left = self.params.crop.as_ref().map(|c| c.left).unwrap_or(0);
        let width = self.params.render_width();
        let height = self.params.render_height();

        let mut res = Vec::new();
        for w in self.params.w_range.clone() {
            for v in (top..top + height).step_by(self.params.tile_size) {
                for u in (left..left + width).step_by(self.params.tile_size) {
                    let tile = UVTile { u, v, w };
                    res.push(tile);
                }
            }
        }
        res
    }

    /// Paint the tile once with a chunk-collecting backend (no data fetched)
    /// to learn exactly which LOD-0 chunks the render will sample.
    fn chunks_for(&self, UVTile { u, v, w }: &UVTile) -> BTreeSet<VolumeChunk> {
        let dummy = Rc::new(TileCollectingVolume::new());
        let width = self.params.width;
        let height = self.params.height;
        let tile_width = self.params.tile_size;
        let tile_height = self.params.tile_size;
        let world = ObjVolume::new(
            self.obj.clone(),
            Volume::from_ref(Arc::new(dummy.as_ref().clone())),
            width,
            height,
        )
        .into_volume();

        let mut image = Image::new(tile_width, tile_height);
        let xyz = [
            *u as i32 + tile_width as i32 / 2,
            *v as i32 + tile_height as i32 / 2,
            *w as i32 - self.params.mid_layer as i32,
        ];
        let config = self.params.drawing_config();
        world.paint(xyz, 0, 1, 2, tile_width, tile_height, 1, 1, &config, &mut image);
        let res = dummy.state.replace(Default::default()).requested_tiles;
        res.into_iter().map(Into::into).collect()
    }

    /// Dispatch every chunk the tile needs into the unified cache and block
    /// until each one is resident (or definitively empty). Re-touching via
    /// `state_or_fetch` on each poll keeps the chunks in the current LRU epoch,
    /// which the purge planner refuses to evict — so a tile's own chunks won't
    /// be purged out from under it between this phase and the render.
    /// Dispatch every chunk the tile needs and wait until each one settles
    /// (Resident/Empty/CooldownMiss — i.e. no longer Pending). The wait is
    /// async (`tokio::time::sleep`) so it occupies no blocking thread; the
    /// render stage's blocking threads stay free for tiles whose chunks are
    /// already resident.
    async fn ensure_resident(&self, chunks: &BTreeSet<VolumeChunk>) {
        let keys: Vec<ChunkKey> = chunks.iter().map(|c| c.key()).collect();

        // Kick off all fetches first so the cache's downloader pool works them
        // concurrently. The initial dispatch can plan/queue a lot of work, so
        // run it on a blocking thread rather than the async executor.
        {
            let cache = self.cache.clone();
            let keys = keys.clone();
            tokio::task::spawn_blocking(move || {
                for key in &keys {
                    let _ = cache.state_or_fetch(*key);
                }
            })
            .await
            .unwrap();
        }

        let mut pending = keys;
        let deadline = Instant::now() + TILE_ENSURE_TIMEOUT;
        while !pending.is_empty() {
            // Keep only the chunks still in flight. Resident/Empty/CooldownMiss
            // are all settled and won't change by waiting longer — in
            // particular CooldownMiss covers out-of-bounds samples (routine at
            // surface edges and along composite normals), so we stop polling
            // those and let the read return 0 there rather than spin to the
            // timeout.
            pending.retain(|key| matches!(self.cache.state_or_fetch(*key).as_ref(), ChunkState::Pending { .. }));
            if pending.is_empty() {
                break;
            }
            if Instant::now() >= deadline {
                log::warn!(
                    "ensure_resident: {} chunk(s) still pending after {:?}; rendering with partial data",
                    pending.len(),
                    TILE_ENSURE_TIMEOUT,
                );
                break;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    }

    fn render_layer_from_tiles(&self, tiles: Vec<(UVTile, Image)>) -> Result<()> {
        let width = self.params.render_width();
        let height = self.params.render_height();
        let left = self.params.render_left();
        let top = self.params.render_top();

        let tile_size = self.params.tile_size;
        let w = tiles[0].0.w;
        let mut image = image::GrayImage::new(width as u32, height as u32);

        // copy in all the tile images
        for (UVTile { u, v, w: _ }, tile_image) in tiles {
            let tile_data = tile_image.data;
            // blit into the right area of image
            for lu in 0..tile_size {
                for lv in 0..tile_size {
                    let gu = u + lu - left;
                    let gv = v + lv - top;
                    if gu < width && gv < height {
                        // edge tiles may spill over boundaries of target image
                        image.put_pixel(gu as u32, gv as u32, Luma([tile_data[lv * tile_size + lu].r() as u8]));
                    }
                }
            }
        }

        image
            .save(format!(
                "{}/{:02}.{}",
                self.params.target_dir, w, self.params.target_format
            ))
            .unwrap();

        Ok(())
    }

    fn render_tile(&self, UVTile { u, v, w }: &UVTile) -> Image {
        let paint_width = self.params.tile_size;
        let paint_height = self.params.tile_size;

        // Fresh per-tile reader over the shared cache: UnifiedVolume holds a
        // thread-local hot slot (!Sync), so each render thread needs its own.
        let vol = UnifiedVolume::new(self.cache.clone()).into_volume();
        let world = ObjVolume::new(self.obj.clone(), vol, self.params.width, self.params.height).into_volume();
        let config = self.params.drawing_config();

        let mut image = Image::new(paint_width, paint_height);
        world.paint(
            [
                *u as i32 + paint_width as i32 / 2,
                *v as i32 + paint_height as i32 / 2,
                *w as i32 - self.params.mid_layer as i32,
            ],
            0,
            1,
            2,
            paint_width,
            paint_height,
            1,
            1,
            &config,
            &mut image,
        );
        image
    }
}

fn resolve_transform(args: &Args) -> Result<Option<AffineTransform>> {
    let Some(t) = args.transform.as_ref() else {
        return Ok(None);
    };
    let mut transform = AffineTransform::from_json_array_or_path(t).map_err(|e| anyhow!("Invalid transform: {}", e))?;
    if args.invert_transform {
        transform = transform
            .invert()
            .map_err(|e| anyhow!("Transform is not invertible: {}", e))?;
    }
    Ok(Some(transform))
}

/// Build a `ChunkCache` backed by an ome-zarr volume, mirroring the GUI's
/// construction chain (see `NewVolumeReference::volume`) but keeping the cache
/// handle so the renderer can dispatch/poll chunks and flush at the end.
fn build_cache(volume_arg: &str, cache_root: PathBuf) -> Result<ChunkCache> {
    let vref = if volume_arg.starts_with("http://") || volume_arg.starts_with("https://") {
        NewVolumeReference::from_url(volume_arg)
    } else {
        NewVolumeReference::from_path(volume_arg)
    }
    .map_err(|e| anyhow!("Could not resolve volume '{}': {}", volume_arg, e))?;

    let (id, location) = match vref {
        NewVolumeReference::OmeZarr { id, location } => (id, location),
        other => {
            return Err(anyhow!(
                "vesuvius-render only supports ome-zarr volumes via the unified cache; '{}' resolved to a different format",
                other.id()
            ))
        }
    };

    let (ome, source_key) = match &location {
        VolumeLocation::RemoteUrl(url) => (OmeZarrContext::from_url_blocking_to_default_cache_dir(url), url.clone()),
        VolumeLocation::LocalPath(path) => (OmeZarrContext::from_path(path), format!("file://{}", path)),
    };

    let unique_id = unified_volume_key(&source_key, &id);
    let native: Arc<dyn ChunkBackfiller> = Arc::new(OmeZarrBackfiller::from_ome(unique_id, ome));
    let backfiller: Arc<dyn ChunkBackfiller> = Arc::new(SynthesizedLodBackfiller::new(native, 32));
    let cache = UnifiedCache::for_cache_dir(cache_root).open_volume(backfiller);
    // The renderer blocks until each chunk is resident before reading, so the
    // upscaled-from-parent preview is never read — skip the per-chunk upsample.
    cache.set_preview_synthesis(false);
    // For the same reason, don't prefetch the coarse preview-LOD pyramid in
    // `touch_aabb`: those chunks are only ever consumed by the GUI's
    // progressive preview / upscale-from-parent, never by the renderer, so
    // fetching+decoding them is pure wasted work (and an accidental LOD climb).
    cache.set_preview_prefetch(false);
    Ok(cache)
}

async fn monitor_runtime_stats(multi: &MultiProgress) {
    let count_style = ProgressStyle::with_template(
        "{spinner} {msg:25} {bar:80.cyan/blue} [{elapsed_precise}] ({eta:>4}) {pos}/{len}",
    )
    .unwrap();
    let bar = ProgressBar::new(0)
        .with_style(count_style)
        .with_message("CPU threads active");

    multi.add(bar.clone());
    tokio::spawn(async move {
        loop {
            let metrics = tokio::runtime::Handle::current().metrics();

            bar.set_length(metrics.num_blocking_threads() as u64);
            bar.set_position((metrics.num_blocking_threads() - metrics.num_idle_blocking_threads()) as u64);

            tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
        }
    });
}

#[derive(Clone)]
struct TileCollectingVolumeState {
    requested_tiles: BTreeSet<(usize, usize, usize)>,
    last_requested: (usize, usize, usize),
}
impl Default for TileCollectingVolumeState {
    fn default() -> Self {
        Self {
            requested_tiles: BTreeSet::new(),
            last_requested: (0, 0, 0),
        }
    }
}

/// A VoxelVolume implementation that just collects needed tiles
#[derive(Clone)]
struct TileCollectingVolume {
    state: RefCell<TileCollectingVolumeState>,
}
impl TileCollectingVolume {
    fn new() -> Self {
        Self {
            state: TileCollectingVolumeState::default().into(),
        }
    }
    fn add_tile(&self, tile: (usize, usize, usize)) {
        let mut state = self.state.borrow_mut();
        if state.last_requested != tile {
            state.last_requested = tile;
            state.requested_tiles.insert(tile);
        }
    }
}
impl PaintVolume for TileCollectingVolume {
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
        panic!();
    }
    fn shared(&self) -> VolumeCons {
        panic!();
    }
}
impl VoxelVolume for TileCollectingVolume {
    fn get(&self, xyz: [f64; 3], _downsampling: i32) -> u8 {
        let tile: (usize, usize, usize) = ((xyz[0] as usize) >> 6, (xyz[1] as usize) >> 6, (xyz[2] as usize) >> 6);

        self.add_tile(tile);
        0
    }
    fn get_interpolated(&self, xyz: [f64; 3], downsampling: i32) -> u8 {
        let x = xyz[0] as usize;
        let y = xyz[1] as usize;
        let z = xyz[2] as usize;

        if x & 63 == 63 || y & 63 == 63 || z & 63 == 63 {
            // slow path, call default trilinear interpolation
            self.get_interpolated_slow(xyz, downsampling);
        } else {
            self.add_tile(((xyz[0] as usize) >> 6, (xyz[1] as usize) >> 6, (xyz[2] as usize) >> 6));
        }
        0
    }
}
