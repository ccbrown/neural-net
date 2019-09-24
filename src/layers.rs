use std::cell::Cell;
use std::rc::Rc;

use ndarray::Dimension;

use super::{algebra, Layer, LayerInstance};

pub struct Flatten {
    pub input_shape: ndarray::IxDyn,
}

impl Layer for Flatten {
    fn init(&self, _namespace: &str) -> Box<LayerInstance> {
        Box::new(FlattenInstance{
            output_size: self.input_shape.size(),
        })
    }

    fn input_shape(&self) -> ndarray::IxDyn {
        self.input_shape.clone()
    }

    fn output_shape(&self) -> ndarray::IxDyn {
        ndarray::Ix1(self.input_shape.size()).into_dyn()
    }
}

pub struct FlattenInstance {
    output_size: usize,
}

impl LayerInstance for FlattenInstance {
    fn expression(&self, input: ndarray::ArrayViewD<algebra::Expr>) -> ndarray::ArrayD<algebra::Expr> {
        input.into_shape(ndarray::Ix1(self.output_size)).unwrap().to_owned().into_dyn()
    }
}

// Dense takes a 1-dimensional input and outputs a 1-dimensional output.
pub struct Dense<Activation, KernelInitializer>
    where Activation: Fn(ndarray::ArrayD<algebra::Expr>) -> ndarray::ArrayD<algebra::Expr> + 'static,
          KernelInitializer: Fn(usize, usize) -> ndarray::Array2<f32>
{
    pub activation: Activation,
    pub kernel_initializer: KernelInitializer,
    pub input_size: usize,
    pub output_size: usize,
}

struct TrainableVariablesBuilder {
    namespace: String,
    trainable_variables: Vec<super::TrainableVariable>,
}

impl TrainableVariablesBuilder {
    fn new<S: Into<String>>(namespace: S) -> Self {
        TrainableVariablesBuilder{
            namespace: namespace.into(),
            trainable_variables: Vec::new(),
        }
    }

    fn append(&mut self, init: f32) -> algebra::Expr {
        let tv = super::TrainableVariable{
            name: format!("{}/v{}", self.namespace, self.trainable_variables.len()),
            value: Rc::new(Cell::new(init)),
        };
        self.trainable_variables.push(tv.clone());
        algebra::v(tv.name, tv.value)
    }
}

impl<Activation, KernelInitializer> Layer for Dense<Activation, KernelInitializer>
    where Activation: Fn(ndarray::ArrayD<algebra::Expr>) -> ndarray::ArrayD<algebra::Expr> + Clone + 'static,
          KernelInitializer: Fn(usize, usize) -> ndarray::Array2<f32>
{
    fn init(&self, namespace: &str) -> Box<LayerInstance> {
        let mut tv_builder = TrainableVariablesBuilder::new(namespace);
        Box::new(DenseInstance{
            activation: self.activation.clone(),
            biases: ndarray::Array::zeros(self.output_size).mapv(|x| tv_builder.append(x)),
            weights: (self.kernel_initializer)(self.output_size, self.input_size).mapv(|x| tv_builder.append(x)),
            trainable_variables: tv_builder.trainable_variables,
        })
    }

    fn input_shape(&self) -> ndarray::IxDyn {
        ndarray::Ix1(self.input_size).into_dyn()
    }

    fn output_shape(&self) -> ndarray::IxDyn {
        ndarray::Ix1(self.output_size).into_dyn()
    }
}

pub struct DenseInstance<Activation>
    where Activation: Fn(ndarray::ArrayD<algebra::Expr>) -> ndarray::ArrayD<algebra::Expr>
{
    activation: Activation,
    biases: ndarray::Array1<algebra::Expr>,
    weights: ndarray::Array2<algebra::Expr>,
    trainable_variables: Vec<super::TrainableVariable>,
}

impl<Activation> LayerInstance for DenseInstance<Activation>
    where Activation: Fn(ndarray::ArrayD<algebra::Expr>) -> ndarray::ArrayD<algebra::Expr>
{
    fn expression(&self, input: ndarray::ArrayViewD<algebra::Expr>) -> ndarray::ArrayD<algebra::Expr> {
        let input = input.into_dimensionality::<ndarray::Ix1>().unwrap();
        let output = algebra::mat_vec_mul(&self.weights, &input) + self.biases.clone();
        (self.activation)(output.into_dyn())
    }

    fn trainable_variables(&self) -> &[super::TrainableVariable] {
        self.trainable_variables.as_slice()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_flatten() {
        let flat = ndarray::Array::range(0.0, 4.0, 1.0);
        let square = flat.clone().into_shape((2, 2)).unwrap();
        assert_eq!(Flatten{
            input_shape: ndarray::Ix2(2, 2).into_dyn(),
        }.init("l").eval(square.into_dyn().view()), flat.into_dyn());
    }

    #[test]
    fn test_dense() {
        let a = ndarray::Array::range(0.0, 16.0, 1.0);
        assert_eq!(Dense{
            activation: |x| x + 1.0,
            kernel_initializer: super::super::initializers::zeros,
            input_size: 16,
            output_size: 4,
        }.init("l").eval(a.into_dyn().view()), ndarray::Array::ones(4).into_dyn());
    }
}
