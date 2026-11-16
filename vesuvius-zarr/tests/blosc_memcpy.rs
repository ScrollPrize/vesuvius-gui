use vesuvius_zarr::blosc::BloscChunk;
use vesuvius_zarr::ZarrArray;

#[test]
fn real_ct_memcpy_chunks_decode_without_a_block_table() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/PHerc0800-blosc");
    let reference = std::fs::read(root.join("reference.raw")).unwrap();
    for compressor in ["lz4", "zstd", "blosclz"] {
        for shuffle in 0..3 {
            let name = format!("{compressor}-shuffle{shuffle}.blosc");
            let bytes = std::fs::read(root.join(&name)).unwrap();
            assert_ne!(bytes[2] & 2, 0);
            assert_eq!(BloscChunk::<u8>::decompress_to_vec(&bytes), reference, "{name}");
        }
    }
}

#[test]
fn real_ct_array_reads_memcpy_chunks() {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/PHerc0800-blosc/volume");
    let reference = include_bytes!("data/PHerc0800-blosc/volume-reference.raw");
    let ctx = ZarrArray::<3, u8>::from_path(path).into_ctx().into_ctx();
    for z in 0..3 {
        for y in 0..16 {
            for x in 0..12 {
                assert_eq!(ctx.get([z, y, x]), Some(reference[(z * 16 + y) * 12 + x]));
            }
        }
    }
}
