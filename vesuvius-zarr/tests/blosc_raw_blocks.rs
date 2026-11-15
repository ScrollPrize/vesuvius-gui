use vesuvius_zarr::blosc::BloscChunk;

#[test]
fn real_ct_frame_with_raw_and_compressed_blocks_decodes_exactly() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/PHerc0800-raw-blocks");
    let reference = std::fs::read(root.join("reference.raw")).unwrap();
    for shuffle in 0..2 {
        let name = format!("lz4-shuffle{shuffle}.blosc");
        let bytes = std::fs::read(root.join(&name)).unwrap();
        assert_eq!(bytes[2] & 2, 0);
        let actual = BloscChunk::<u8>::decompress_to_vec(&bytes);
        assert_eq!(actual.len(), reference.len());
        let wrong = actual.iter().zip(&reference).filter(|(a, b)| a != b).count();
        assert_eq!(wrong, 0, "{name} changed {wrong} real CT bytes");
    }
}
