use vesuvius_zarr::ZarrArray;

fn context() -> vesuvius_zarr::ZarrContext<3> {
    ZarrArray::<3, u8>::from_path(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/PHerc0800"))
        .into_ctx()
        .into_ctx()
}

#[test]
fn integer_coordinates_use_exclusive_volume_bounds() {
    let ctx = context();
    assert!(ctx.get([2, 31, 31]).is_some());
    for axis in 0..3 {
        for value in [ctx.shape()[axis], ctx.shape()[axis] + 1, usize::MAX] {
            let mut point = [1, 10, 10];
            point[axis] = value;
            assert_eq!(ctx.get(point), None, "{point:?}");
        }
    }
}

#[test]
fn interpolation_rejects_invalid_coordinates_and_incomplete_neighborhoods() {
    let ctx = context();
    for axis in 0..3 {
        for value in [
            -0.5,
            -1.0,
            f64::NAN,
            f64::INFINITY,
            f64::NEG_INFINITY,
            ctx.shape()[axis] as f64 - 1.0,
            ctx.shape()[axis] as f64,
            f64::MAX,
        ] {
            let mut point = [0.5, 10.5, 10.5];
            point[axis] = value;
            assert_eq!(ctx.get_interpolated(point), None, "{point:?}");
        }
    }
}

#[test]
fn valid_interpolation_matches_the_real_ct_neighbors() {
    let ctx = context();
    let bytes = include_bytes!("data/PHerc0800/0.0.0");
    for point in [[0.25, 5.5, 6.75], [1.5, 30.25, 30.75]] {
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
                    let index = ((starts[0] + z) * 32 + starts[1] + y) * 32 + starts[2] + x;
                    expected += bytes[index] as f64 * weight;
                }
            }
        }
        assert_eq!(ctx.get_interpolated(point), Some(expected as u8), "{point:?}");
    }
}
