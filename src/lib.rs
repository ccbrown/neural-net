extern crate byteorder;
#[macro_use] extern crate log;
extern crate ndarray;
extern crate rand;
extern crate reqwest;
#[macro_use] extern crate simple_error;

use std::error::Error;
use std::rc::Rc;

#[derive(Clone)]
pub struct LayerVariable {
    pub name: String,
    pub value: Rc<algebra::VariableValue>,
}

pub trait Layer {
    fn init(&self, namespace: &str) -> Box<LayerInstance>;

    fn input_shape(&self) -> ndarray::IxDyn;

    fn output_shape(&self) -> ndarray::IxDyn;
}

pub trait LayerInstance {
    fn eval(&self, input: ndarray::ArrayViewD<f32>) -> ndarray::ArrayD<f32> {
        self.expression(algebra::expr(input)).eval()
    }

    fn expression(&self, input: algebra::Expr) -> algebra::Expr;

    fn variables(&self) -> &[LayerVariable] {
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
pub mod losses;
pub mod models;
pub mod util;
