use vesuvius_zarr::ZarrArray;

#[test]
fn explicit_blosc_block_sizes_preserve_real_ct_voxels() {
    let reference = include_bytes!("data/PHerc0800-blocksize/reference.raw");
    for blocksize in [0, 65536, 131072] {
        let path = format!(
            "{}/tests/data/PHerc0800-blocksize/{blocksize}",
            env!("CARGO_MANIFEST_DIR")
        );
        let ctx = ZarrArray::<3, u8>::from_path(&path).into_ctx().into_ctx();
        for z in 0..3 {
            for y in 0..32 {
                for x in 0..32 {
                    assert_eq!(
                        ctx.get([z, y, x]),
                        Some(reference[(z * 32 + y) * 32 + x]),
                        "blocksize={blocksize} at {z},{y},{x}"
                    );
                }
            }
        }
    }
}
