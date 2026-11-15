use vesuvius_zarr::blosc::BloscChunk;

#[test]
fn real_ct_bitshuffle_is_reversed_after_decompression() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/PHerc0800-bitshuffle");
    for name in ["lz4-8192", "zstd-8192", "zstd-8197"] {
        let reference = std::fs::read(root.join(format!("{name}.raw"))).unwrap();
        let encoded = std::fs::read(root.join(format!("{name}.blosc"))).unwrap();
        let decoded = BloscChunk::<u8>::decompress_to_vec(&encoded);
        assert_eq!(decoded.len(), reference.len(), "{name}");
        let wrong = decoded.iter().zip(&reference).filter(|(a, b)| a != b).count();
        assert_eq!(wrong, 0, "{name}: {wrong} real CT bytes changed");
    }
}
