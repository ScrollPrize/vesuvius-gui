//! Streaming tiled-BigTIFF writer for very large single-channel (Gray8)
//! renders.
//!
//! The `tiff` crate's encoder only writes *strip*-organized files, and its
//! high-level `ImageEncoder` wants the whole image (or whole strips) in memory
//! at once. For an ~11 GPixel layer that means buffering a full layer of
//! `Color32` tiles (~4 B/px) plus a full `GrayImage` (~1 B/px) — tens of GiB —
//! before a single byte hits disk.
//!
//! A rendered layer is already a grid of `tile_size × tile_size` tiles, which
//! is exactly the on-disk layout of a *tiled* TIFF. So we write the tiled
//! layout by hand via the low-level [`DirectoryEncoder`]: each tile's pixels
//! are flushed to the file as soon as they're produced (in any order), and only
//! the small `TileOffsets`/`TileByteCounts` arrays are held until `finish`.
//! Peak memory becomes O(a few in-flight tiles) instead of O(whole layer).
//!
//! TIFF stores edge tiles at *full* tile size with padding the reader clips to
//! `ImageWidth`/`ImageLength`, which matches how the renderer paints full tiles
//! that overhang the image — so tiles are written verbatim, no clipping.

use anyhow::{bail, Result};
use std::fs::File;
use std::io::BufWriter;
use tiff::encoder::{TiffEncoder, TiffKind, TiffKindBig, TiffKindStandard};
use tiff::tags::Tag;

/// One tile's Gray8 pixels, row-major and padded to the full
/// `tile_size × tile_size` (edge tiles included).
pub struct TileGray {
    /// Tile column, `0..width.div_ceil(tile_size)`.
    pub col: usize,
    /// Tile row, `0..height.div_ceil(tile_size)`.
    pub row: usize,
    /// `tile_size * tile_size` gray bytes.
    pub data: Vec<u8>,
}

/// Write a tiled Gray8 TIFF at `path`, consuming `tiles` in any order. Picks
/// BigTIFF once the pixel data would cross 2 GiB (classic TIFF's 32-bit offsets
/// overflow — and many readers treat them as signed, breaking at 2 GiB).
///
/// Errors (and drops the partial file's fate to the caller) if `tile_size`
/// isn't a multiple of 16 (a TIFF tiling requirement), if any tile is
/// mis-sized/out-of-range, or if the stream ends before every tile of the
/// `width × height` grid was seen.
pub fn write_tiled_gray_tiff(
    path: &str,
    width: usize,
    height: usize,
    tile_size: usize,
    metadata: &str,
    tiles: impl Iterator<Item = TileGray>,
) -> Result<()> {
    if tile_size == 0 || tile_size % 16 != 0 {
        bail!("tiled TIFF requires tile_size to be a positive multiple of 16, got {tile_size}");
    }

    // 2 GiB. Threshold on the *pixel* data; the tile offset/count arrays and
    // tags add only a few hundred KiB, so this keeps us clear of both the
    // 4 GiB hard limit and the 2 GiB signed-offset limit.
    const BIGTIFF_THRESHOLD: u64 = 2 * 1024 * 1024 * 1024;
    let pixel_bytes = width as u64 * height as u64; // Gray8 = 1 B/px

    let file = BufWriter::new(File::create(path)?);
    if pixel_bytes >= BIGTIFF_THRESHOLD {
        encode::<TiffKindBig>(file, width, height, tile_size, metadata, tiles)
    } else {
        encode::<TiffKindStandard>(file, width, height, tile_size, metadata, tiles)
    }
}

fn encode<K: TiffKind>(
    file: BufWriter<File>,
    width: usize,
    height: usize,
    tile_size: usize,
    metadata: &str,
    tiles: impl Iterator<Item = TileGray>,
) -> Result<()> {
    let tiles_across = width.div_ceil(tile_size);
    let tiles_down = height.div_ceil(tile_size);
    let n = tiles_across * tiles_down;
    let expected_len = tile_size * tile_size;

    let mut encoder = TiffEncoder::<_, K>::new_generic(file)?;
    let mut dir = encoder.image_directory()?;

    // Data is appended to the file as it arrives; the offset/count of each tile
    // is recorded at its grid index so the tag arrays come out in tile order
    // regardless of arrival order. Kept as raw u64/u32 and narrowed to the
    // kind's offset width only at the end.
    let mut offsets = vec![0u64; n];
    let mut byte_counts = vec![0u32; n];
    let mut seen = vec![false; n];
    let mut written = 0usize;

    for TileGray { col, row, data } in tiles {
        if data.len() != expected_len {
            bail!(
                "tile ({col},{row}) has {} bytes, expected {expected_len} ({tile_size}x{tile_size})",
                data.len()
            );
        }
        if col >= tiles_across || row >= tiles_down {
            bail!("tile ({col},{row}) out of range {tiles_across}x{tiles_down}");
        }
        let idx = row * tiles_across + col;
        let offset = dir.write_data(data.as_slice())?;
        offsets[idx] = offset;
        byte_counts[idx] = data.len() as u32;
        if !seen[idx] {
            seen[idx] = true;
            written += 1;
        }
    }

    if written != n {
        bail!("tiled TIFF incomplete: wrote {written} of {n} tiles for {width}x{height} image");
    }

    let offsets: Vec<K::OffsetType> = offsets
        .into_iter()
        .map(K::convert_offset)
        .collect::<tiff::TiffResult<Vec<_>>>()?;

    // Baseline tags for a tiled, uncompressed, single-sample grayscale image.
    dir.write_tag(Tag::ImageWidth, width as u32)?;
    dir.write_tag(Tag::ImageLength, height as u32)?;
    dir.write_tag(Tag::BitsPerSample, 8u16)?;
    dir.write_tag(Tag::Compression, 1u16)?; // none
    dir.write_tag(Tag::PhotometricInterpretation, 1u16)?; // BlackIsZero
    dir.write_tag(Tag::SamplesPerPixel, 1u16)?;
    dir.write_tag(Tag::PlanarConfiguration, 1u16)?;
    dir.write_tag(Tag::TileWidth, tile_size as u32)?;
    dir.write_tag(Tag::TileLength, tile_size as u32)?;
    dir.write_tag(Tag::TileOffsets, K::convert_slice(&offsets))?;
    dir.write_tag(Tag::TileByteCounts, byte_counts.as_slice())?;
    dir.write_tag(Tag::ImageDescription, metadata)?;

    dir.finish()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tiff::decoder::{Decoder, DecodingResult};

    fn pixel(gx: usize, gy: usize) -> u8 {
        ((gx * 7 + gy * 13) % 251) as u8
    }

    /// Build the full tile set for a `width × height` grid (edge tiles padded
    /// to full `tile_size`), each pixel stamped with `pixel(gx, gy)`.
    fn tiles(width: usize, height: usize, tile_size: usize) -> Vec<TileGray> {
        let across = width.div_ceil(tile_size);
        let down = height.div_ceil(tile_size);
        let mut out = Vec::new();
        for row in 0..down {
            for col in 0..across {
                let mut data = vec![0u8; tile_size * tile_size];
                for ly in 0..tile_size {
                    for lx in 0..tile_size {
                        data[ly * tile_size + lx] = pixel(col * tile_size + lx, row * tile_size + ly);
                    }
                }
                out.push(TileGray { col, row, data });
            }
        }
        out
    }

    fn roundtrip_check(path: &std::path::Path, width: usize, height: usize) {
        let mut dec = Decoder::new(std::io::BufReader::new(File::open(path).unwrap())).unwrap();
        assert_eq!(dec.dimensions().unwrap(), (width as u32, height as u32));
        let img = dec.read_image().unwrap();
        let data = match img {
            DecodingResult::U8(v) => v,
            other => panic!("expected U8, got {:?}", std::mem::discriminant(&other)),
        };
        assert_eq!(data.len(), width * height, "decoded sample count clipped to image size");
        for gy in 0..height {
            for gx in 0..width {
                assert_eq!(data[gy * width + gx], pixel(gx, gy), "mismatch at ({gx},{gy})");
            }
        }
    }

    #[test]
    fn ragged_edges_roundtrip() {
        // Neither dimension is a multiple of the tile size, so both the right
        // column and bottom row of tiles are padded and must clip on readback.
        let (w, h, ts) = (40usize, 24usize, 16usize);
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("layer.tif");
        let mut t = tiles(w, h, ts);
        // Feed out of order — offsets must still land at the right grid index.
        t.reverse();
        write_tiled_gray_tiff(path.to_str().unwrap(), w, h, ts, "meta", t.into_iter()).unwrap();
        roundtrip_check(&path, w, h);
    }

    #[test]
    fn bigtiff_offsets_roundtrip() {
        // Exercise the 64-bit-offset encode path directly (the size-based
        // switch would need >2 GiB of pixels to trigger organically).
        let (w, h, ts) = (48usize, 32usize, 16usize);
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("big.tif");
        let file = BufWriter::new(File::create(&path).unwrap());
        encode::<TiffKindBig>(file, w, h, ts, "meta", tiles(w, h, ts).into_iter()).unwrap();
        roundtrip_check(&path, w, h);
    }

    #[test]
    fn rejects_unaligned_tile_size() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("bad.tif");
        let err = write_tiled_gray_tiff(path.to_str().unwrap(), 20, 20, 20, "m", std::iter::empty());
        assert!(err.is_err(), "tile_size not a multiple of 16 must be rejected");
    }

    #[test]
    fn rejects_missing_tiles() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("short.tif");
        // 3x2 grid but only feed one tile.
        let one = tiles(40, 24, 16).into_iter().take(1);
        let err = write_tiled_gray_tiff(path.to_str().unwrap(), 40, 24, 16, "m", one);
        assert!(err.is_err(), "incomplete tile set must be rejected");
    }
}
