use vesuvius_zarr::ZarrArray;

fn context(order: &str) -> vesuvius_zarr::ZarrContext<3> {
    let path = format!("{}/tests/data/PHerc0800-order/{order}", env!("CARGO_MANIFEST_DIR"));
    ZarrArray::<3, u8>::from_path(&path).into_ctx().into_ctx()
}

#[test]
fn real_ct_voxels_are_independent_of_storage_order() {
    let reference = include_bytes!("data/PHerc0800-order/reference.raw");
    for order in ["C", "F"] {
        let ctx = context(order);
        for z in 0..3 {
            for y in 0..16 {
                for x in 0..12 {
                    assert_eq!(
                        ctx.get([z, y, x]),
                        Some(reference[(z * 16 + y) * 12 + x]),
                        "{order} at {z},{y},{x}"
                    );
                }
            }
        }
    }
}

#[test]
fn real_ct_interpolation_is_independent_of_storage_order() {
    let reference = include_bytes!("data/PHerc0800-order/reference.raw");
    for order in ["C", "F"] {
        let ctx = context(order);
        for point in [[0.25, 3.25, 3.75], [1.25, 7.25, 5.75]] {
            let starts = point.map(|v| v as usize);
            let fractions = std::array::from_fn::<_, 3, _>(|axis| point[axis] - starts[axis] as f64);
            let mut expected = 0.0;
            for z in 0..2 {
                for y in 0..2 {
                    for x in 0..2 {
                        let sides = [z, y, x];
                        let weight: f64 = (0..3)
                            .map(|axis| {
                                if sides[axis] == 0 {
                                    1.0 - fractions[axis]
                                } else {
                                    fractions[axis]
                                }
                            })
                            .product();
                        let index = ((starts[0] + z) * 16 + starts[1] + y) * 12 + starts[2] + x;
                        expected += reference[index] as f64 * weight;
                    }
                }
            }
            assert_eq!(
                ctx.get_interpolated(point),
                Some(expected as u8),
                "{order} at {point:?}"
            );
        }
    }
}
