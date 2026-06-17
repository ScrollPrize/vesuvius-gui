//! Render provenance: builds a JSON blob describing how an artifact was
//! produced (build version + the parameters it ran with) and embeds it inside
//! each output file's native metadata — a PNG `tEXt` chunk, a TIFF
//! `ImageDescription` tag, or a JPEG `COM` comment segment. The blob survives a
//! plain copy of the image, so a stray layer image can always be traced back to
//! the exact build and command that made it.

use crate::Args;
use anyhow::Result;
use image::{GrayImage, ImageEncoder};
use serde::Serialize;

/// Git revision of the build (suffixed `-dirty` for a modified working tree),
/// baked in by `build.rs`. `"unknown"` if git was unavailable at build time.
pub const GIT_REVISION: &str = env!("VESUVIUS_GIT_REVISION");
/// UTC build timestamp (ISO-8601), baked in by `build.rs`.
pub const BUILD_TIME: &str = env!("VESUVIUS_BUILD_TIME");

/// The keyword/identifier under which the JSON blob is stored. Used as the PNG
/// text-chunk keyword (Latin-1, <=79 chars).
const METADATA_KEYWORD: &str = "vesuvius-render";

/// Everything we record about a render. Serialized once and embedded verbatim
/// into every layer image of the run.
#[derive(Serialize)]
struct RenderMetadata<'a> {
    tool: &'static str,
    tool_version: &'static str,
    git_revision: &'static str,
    build_time: &'static str,
    /// The raw process argv, exactly as invoked.
    command_line: Vec<String>,
    /// The parsed parameters (a `None` field means the flag was not passed).
    params: &'a Args,
    /// Identity of the segment source files actually read (size + content hash).
    source: SourceInfo,
}

/// Identifies the rendered segment by the source files it was read from.
#[derive(Serialize)]
struct SourceInfo {
    /// `"obj"`, `"tifxyz"`, or `"none"`.
    kind: &'static str,
    files: Vec<SourceFile>,
}

/// A single source file's identity: where it is, how big it is, and its
/// SHA-256, so a render can be matched back to the exact input bytes.
#[derive(Serialize)]
struct SourceFile {
    /// Canonical (absolute) path when resolvable, else the path as given.
    path: String,
    /// Size in bytes, or `None` if the file could not be stat'd.
    bytes: Option<u64>,
    /// Lowercase hex SHA-256 of the file contents, or `None` if unreadable.
    sha256: Option<String>,
}

/// Stat + hash a single file. Best-effort: missing/unreadable files yield
/// `None` for the affected fields rather than failing the render.
fn describe_file(path: &str) -> SourceFile {
    let p = std::path::Path::new(path);
    let canonical = std::fs::canonicalize(p)
        .map(|c| c.display().to_string())
        .unwrap_or_else(|_| path.to_string());
    let bytes = std::fs::metadata(p).ok().map(|m| m.len());
    SourceFile {
        path: canonical,
        bytes,
        sha256: sha256_file(p),
    }
}

/// Stream a file through SHA-256 without loading it fully into memory.
fn sha256_file(path: &std::path::Path) -> Option<String> {
    use sha2::{Digest, Sha256};
    use std::io::Read;
    let mut file = std::fs::File::open(path).ok()?;
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 1 << 16];
    loop {
        match file.read(&mut buf) {
            Ok(0) => break,
            Ok(n) => hasher.update(&buf[..n]),
            Err(_) => return None,
        }
    }
    Some(format!("{:x}", hasher.finalize()))
}

/// Collect identity for the segment source: the `.obj` file, or the tifxyz
/// directory's `meta.json` + `x/y/z.tif` grid.
fn build_source_info(args: &Args) -> SourceInfo {
    if let Some(obj) = args.obj_path() {
        SourceInfo {
            kind: "obj",
            files: vec![describe_file(obj)],
        }
    } else if let Some(dir) = args.tifxyz_path() {
        let files = ["meta.json", "x.tif", "y.tif", "z.tif"]
            .iter()
            .map(|name| describe_file(&format!("{}/{}", dir.trim_end_matches('/'), name)))
            .collect();
        SourceInfo { kind: "tifxyz", files }
    } else {
        SourceInfo {
            kind: "none",
            files: vec![],
        }
    }
}

/// Serialize the provenance for `args` into a pretty JSON string, to be embedded
/// in every output file of this run.
pub fn build_metadata_json(args: &Args) -> String {
    let metadata = RenderMetadata {
        tool: "vesuvius-render",
        tool_version: env!("CARGO_PKG_VERSION"),
        git_revision: GIT_REVISION,
        build_time: BUILD_TIME,
        command_line: std::env::args().collect(),
        params: args,
        source: build_source_info(args),
    };
    // Serialization can only fail on a non-serializable type, which is a
    // compile-time property of `Args`; fall back rather than abort a render.
    serde_json::to_string_pretty(&metadata).unwrap_or_else(|e| format!("{{\"error\":\"{}\"}}", e))
}

/// Save `image` to `path`, embedding `metadata` in the file's native metadata
/// slot. `format` is the (case-insensitive) target extension. Unknown formats
/// fall back to a plain save (no metadata).
pub fn save_image_with_metadata(image: &GrayImage, path: &str, format: &str, metadata: &str) -> Result<()> {
    match format.to_ascii_lowercase().as_str() {
        "png" => save_png(image, path, metadata),
        "tif" | "tiff" => save_tiff(image, path, metadata),
        "jpg" | "jpeg" => save_jpeg(image, path, metadata),
        _ => {
            image.save(path)?;
            Ok(())
        }
    }
}

fn save_png(image: &GrayImage, path: &str, metadata: &str) -> Result<()> {
    let file = std::io::BufWriter::new(std::fs::File::create(path)?);
    let mut encoder = png::Encoder::new(file, image.width(), image.height());
    encoder.set_color(png::ColorType::Grayscale);
    encoder.set_depth(png::BitDepth::Eight);
    // A `zTXt` chunk compresses the (repetitive) JSON; harmless if it fails.
    encoder.add_ztxt_chunk(METADATA_KEYWORD.to_string(), metadata.to_string())?;
    let mut writer = encoder.write_header()?;
    writer.write_image_data(image.as_raw())?;
    writer.finish()?;
    Ok(())
}

fn save_tiff(image: &GrayImage, path: &str, metadata: &str) -> Result<()> {
    use tiff::encoder::{colortype::Gray8, TiffEncoder};
    use tiff::tags::Tag;

    let file = std::io::BufWriter::new(std::fs::File::create(path)?);
    let mut tiff = TiffEncoder::new(file)?;
    let mut layer = tiff.new_image::<Gray8>(image.width(), image.height())?;
    layer.encoder().write_tag(Tag::ImageDescription, metadata)?;
    layer.write_data(image.as_raw())?;
    Ok(())
}

fn save_jpeg(image: &GrayImage, path: &str, metadata: &str) -> Result<()> {
    // The `image` JPEG encoder exposes no comment API, so encode to memory and
    // splice a standard `COM` (0xFFFE) segment in right after the SOI marker.
    let mut buf: Vec<u8> = Vec::new();
    image::codecs::jpeg::JpegEncoder::new(&mut buf).write_image(
        image.as_raw(),
        image.width(),
        image.height(),
        image::ExtendedColorType::L8,
    )?;
    std::fs::write(path, inject_jpeg_comment(buf, metadata.as_bytes()))?;
    Ok(())
}

/// Insert a JPEG `COM` comment segment carrying `comment` immediately after the
/// SOI marker. Returns the input unchanged if it isn't a recognizable JPEG. A
/// `COM` segment's length field is 16-bit, so the comment is truncated to fit.
fn inject_jpeg_comment(jpeg: Vec<u8>, comment: &[u8]) -> Vec<u8> {
    if jpeg.len() < 2 || jpeg[0] != 0xFF || jpeg[1] != 0xD8 {
        return jpeg;
    }
    // The length field counts itself (2 bytes) plus the payload, max 0xFFFF.
    let payload = &comment[..comment.len().min(0xFFFF - 2)];
    let len = (payload.len() + 2) as u16;
    let mut out = Vec::with_capacity(jpeg.len() + payload.len() + 4);
    out.extend_from_slice(&jpeg[..2]); // SOI
    out.extend_from_slice(&[0xFF, 0xFE]); // COM marker
    out.extend_from_slice(&len.to_be_bytes());
    out.extend_from_slice(payload);
    out.extend_from_slice(&jpeg[2..]); // remainder of the stream
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Each format must (a) embed the metadata so it's recoverable from the
    /// file bytes and (b) still decode back to the original pixels.
    fn roundtrip(format: &str) {
        let mut image = GrayImage::new(8, 4);
        for (i, p) in image.pixels_mut().enumerate() {
            p.0[0] = (i * 7) as u8;
        }
        let meta = "{\"git_revision\":\"abc123\",\"hello\":\"world\"}";
        let dir = std::env::temp_dir();
        let path = dir.join(format!("vesuvius_meta_test.{}", format));
        let path = path.to_str().unwrap();

        save_image_with_metadata(&image, path, format, meta).unwrap();

        let bytes = std::fs::read(path).unwrap();
        assert!(
            bytes.windows(meta.len()).any(|w| w == meta.as_bytes())
                // PNG zTXt compresses the text, so the literal won't appear;
                // assert the keyword chunk is present instead.
                || (format == "png" && bytes.windows(METADATA_KEYWORD.len()).any(|w| w == METADATA_KEYWORD.as_bytes())),
            "metadata not embedded in {} output",
            format
        );

        let decoded = image::open(path).unwrap().to_luma8();
        assert_eq!(decoded.dimensions(), image.dimensions());
        assert_eq!(decoded.as_raw(), image.as_raw(), "{} pixels altered", format);

        std::fs::remove_file(path).ok();
    }

    #[test]
    fn png_embeds_metadata_and_preserves_pixels() {
        roundtrip("png");
    }

    #[test]
    fn tiff_embeds_metadata_and_preserves_pixels() {
        roundtrip("tiff");
    }

    #[test]
    fn jpeg_embeds_comment() {
        // JPEG is lossy, so only assert the COM segment carries the metadata.
        let image = GrayImage::new(8, 4);
        let meta = "{\"git_revision\":\"abc123\"}";
        let path = std::env::temp_dir().join("vesuvius_meta_test_comment.jpg");
        let path = path.to_str().unwrap();
        save_image_with_metadata(&image, path, "jpeg", meta).unwrap();
        let bytes = std::fs::read(path).unwrap();
        assert_eq!(&bytes[..2], &[0xFF, 0xD8], "not a JPEG");
        assert_eq!(&bytes[2..4], &[0xFF, 0xFE], "COM segment not right after SOI");
        assert!(bytes.windows(meta.len()).any(|w| w == meta.as_bytes()));
        image::open(path).unwrap(); // still decodable
        std::fs::remove_file(path).ok();
    }
}
