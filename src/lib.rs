extern crate byteorder;
extern crate ndarray;
extern crate rand;
extern crate reqwest;
#[macro_use] extern crate simple_error;

use std::error::Error;
use std::fmt;
use std::rc::Rc;

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

#[derive(Clone)]
pub struct TrainableVariable {
    pub name: String,
    pub value: Rc<f32>,
}

pub trait Layer {
    type Instance;

    fn init(&self, namespace: &str) -> Self::Instance;

    fn input_shape(&self) -> Shape;

    fn output_shape(&self) -> Shape;
}

pub trait LayerInstance {
    fn eval<'a, D: ndarray::Dimension>(&self, input: ndarray::ArrayView<'a, f32, D>) -> ndarray::ArrayD<f32>
        where D: ndarray::Dimension
    {
        self.expression(input.mapv(algebra::c).view()).mapv(|e| e.eval())
    }

    fn expression<'a, D: ndarray::Dimension>(&self, input: ndarray::ArrayView<'a, algebra::Expr, D>) -> ndarray::ArrayD<algebra::Expr>;

    fn trainable_variables(&self) -> &[TrainableVariable] {
        &[]
    }
}

pub trait Dataset {
    fn len(&self) -> usize;

    fn input(&mut self, i: usize) -> Result<ndarray::ArrayViewD<f32>, Box<Error>>;

    fn target(&mut self, i: usize) -> Result<ndarray::ArrayViewD<f32>, Box<Error>>;
}

pub mod algebra;
pub mod activations;
pub mod datasets;
pub mod initializers;
pub mod layers;
pub mod models;
pub mod util;
