use rand::distributions::{Distribution, Uniform};
use rand::SeedableRng;

use ndarray::Dimension;

pub fn glorot_uniform(shape: &ndarray::IxDyn) -> ndarray::ArrayD<f32> {
    let shape_arr = shape.as_array_view();
    let fan_in_plus_out = match shape_arr.len() {
        2 => shape_arr.sum(),
        3 => shape_arr[0] * (shape_arr[1] + shape_arr[2]),
        4 => shape_arr[0] * shape_arr[1] * (shape_arr[2] + shape_arr[3]),
        _ => panic!("unsupported shape dimensionality for glorot uniform"),
    };
    let limit = (6.0 / fan_in_plus_out as f32).sqrt();
    let mut rng = rand::rngs::StdRng::seed_from_u64(0);
    let dist = Uniform::new_inclusive(-limit, limit);
    ndarray::Array::from_shape_fn(shape.clone(), |_| dist.sample(&mut rng))
}

pub fn zeros(shape: &ndarray::IxDyn) -> ndarray::ArrayD<f32> {
    ndarray::Array::zeros(shape.clone())
}

pub fn ones(shape: &ndarray::IxDyn) -> ndarray::ArrayD<f32> {
    ndarray::Array::ones(shape.clone())
}

pub fn copy(v: ndarray::ArrayD<f32>) -> impl Fn(&ndarray::IxDyn) -> ndarray::ArrayD<f32> {
    move |shape| {
        if *shape != v.dim() {
            panic!("differing shapes for copy initializer: got {:?}, expected {:?}", shape, v.dim());
        }
        v.clone()
    }
}
