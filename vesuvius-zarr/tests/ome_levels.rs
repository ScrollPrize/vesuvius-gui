use vesuvius_zarr::OmeZarrContext;

fn context(name: &str) -> OmeZarrContext {
    OmeZarrContext::from_path(&format!(
        "{}/tests/data/PHerc0800-levels/{name}",
        env!("CARGO_MANIFEST_DIR")
    ))
}

#[test]
fn missing_middle_level_preserves_coarse_ct_coordinates() {
    let ctx = context("middle-missing");
    let coarse = include_bytes!("data/PHerc0800-levels/reference-2.raw");
    for scale in [1, 2] {
        for z in 0..20 {
            for y in 0..32 {
                for x in 0..32 {
                    assert_eq!(
                        ctx.get([z, y, x], scale),
                        coarse[((z >> 2) * 8 + (y >> 2)) * 8 + (x >> 2)],
                        "scale={scale} at {z},{y},{x}"
                    );
                }
            }
        }
    }
}

#[test]
fn missing_base_level_keeps_its_original_downsampling() {
    let ctx = context("base-missing");
    let reference = include_bytes!("data/PHerc0800-levels/reference-1.raw");
    for z in 0..20 {
        for y in 0..32 {
            for x in 0..32 {
                assert_eq!(
                    ctx.get([z, y, x], 0),
                    reference[((z >> 1) * 16 + (y >> 1)) * 16 + (x >> 1)]
                );
            }
        }
    }
}

#[test]
fn complete_pyramid_keeps_all_resolutions() {
    let ctx = context("complete");
    let references: [&[u8]; 3] = [
        include_bytes!("data/PHerc0800-levels/reference-0.raw"),
        include_bytes!("data/PHerc0800-levels/reference-1.raw"),
        include_bytes!("data/PHerc0800-levels/reference-2.raw"),
    ];
    for scale in 0..3 {
        let side = 32 >> scale;
        for z in 0..20 {
            for y in 0..32 {
                for x in 0..32 {
                    assert_eq!(
                        ctx.get([z, y, x], scale),
                        references[scale as usize][((z >> scale) * side + (y >> scale)) * side + (x >> scale)]
                    );
                }
            }
        }
    }
}

#[test]
fn group_without_local_levels_reads_as_empty() {
    let ctx = context("empty");
    assert_eq!(ctx.get([0, 0, 0], 0), 0);
    assert_eq!(ctx.get_interpolated([0.0, 0.0, 0.0], 0), 0);
}

#[test]
fn missing_levels_preserve_interpolated_ct_coordinates() {
    let complete = context("complete");
    for (name, scale, expected_scale) in [("middle-missing", 1, 2), ("base-missing", 0, 1)] {
        let partial = context(name);
        for point in [[0.5, 0.75, 3.25], [6.5, 12.25, 20.5], [11.75, 19.0, 18.5]] {
            assert_eq!(
                partial.get_interpolated(point, scale),
                complete.get_interpolated(point, expected_scale),
                "{name} at {point:?}"
            );
        }
    }
}
