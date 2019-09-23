extern crate byteorder;
extern crate ndarray;
extern crate rand;
extern crate reqwest;
#[macro_use] extern crate simple_error;

use std::error::Error;
use std::fmt;

#[derive(Clone, Copy, PartialEq)]
pub enum Shape {
    D1(usize),
    D2(usize, usize),
}

impl Shape {
    pub fn size(&self) -> usize {
        match *self {
            Shape::D1(x) => x,
            Shape::D2(x, y) => x * y,
        }
    }
}

impl fmt::Display for Shape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Shape::D1(x) => write!(f, "({})", x),
            Shape::D2(x, y) => write!(f, "({}, {})", x, y),
        }
    }
}

pub trait Layer {
    type Instance;

    fn init(&self) -> Self::Instance;

    fn input_shape(&self) -> Shape;

    fn output_shape(&self) -> Shape;
}

pub trait LayerInstance {
    fn eval<'a, D>(&self, input: ndarray::ArrayView<'a, f32, D>) -> Result<ndarray::ArrayD<f32>, Box<Error>>
        where D: ndarray::Dimension
    ;
}

pub trait Dataset {
    fn len(&self) -> usize;

    fn input(&mut self, i: usize) -> Result<ndarray::ArrayViewD<f32>, Box<Error>>;

    fn target(&mut self, i: usize) -> Result<ndarray::ArrayViewD<f32>, Box<Error>>;
}

pub mod activations;
pub mod datasets;
pub mod initializers;
pub mod layers;
pub mod models;
pub mod util;
