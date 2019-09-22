use rand::distributions::{Distribution, Uniform};
use rand::SeedableRng;

pub fn glorot_uniform(rows: usize, cols: usize) -> ndarray::Array2<f32> {
    let limit = (6.0 / (rows + cols) as f32).sqrt();
    let mut rng = rand::rngs::StdRng::seed_from_u64(0);
    let dist = Uniform::new_inclusive(-limit, limit);
    ndarray::Array::from_shape_fn((rows, cols), |_| dist.sample(&mut rng))
}

pub fn zeros(rows: usize, cols: usize) -> ndarray::Array2<f32> {
    ndarray::Array::zeros((rows, cols))
}

pub fn ones(rows: usize, cols: usize) -> ndarray::Array2<f32> {
    ndarray::Array::ones((rows, cols))
}
