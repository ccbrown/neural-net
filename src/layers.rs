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
    fn expression(&self, input: algebra::Expr) -> algebra::Expr {
        input.reshape(ndarray::Ix1(self.output_size))
    }
}

// Dense takes a 1-dimensional input and outputs a 1-dimensional output.
pub struct Dense<Activation, KernelInitializer>
    where Activation: Fn(algebra::Expr) -> algebra::Expr + 'static,
          KernelInitializer: Fn(usize, usize) -> ndarray::Array2<f32>
{
    pub activation: Activation,
    pub kernel_initializer: KernelInitializer,
    pub input_size: usize,
    pub output_size: usize,
}

struct LayerVariablesBuilder {
    namespace: String,
    variables: Vec<super::LayerVariable>,
}

impl LayerVariablesBuilder {
    fn new<S: Into<String>>(namespace: S) -> Self {
        LayerVariablesBuilder{
            namespace: namespace.into(),
            variables: Vec::new(),
        }
    }

    fn append<S1, D>(&mut self, init: ndarray::ArrayBase<S1, D>) -> algebra::Expr
        where S1: ndarray::Data<Elem=f32>,
              D: ndarray::Dimension,
    {
        let v = super::LayerVariable{
            name: format!("{}.v{}", self.namespace, self.variables.len()),
            value: Rc::new(algebra::VariableValue::new(init)),
        };
        self.variables.push(v.clone());
        algebra::v(v.name, v.value)
    }
}

impl<Activation, KernelInitializer> Layer for Dense<Activation, KernelInitializer>
    where Activation: Fn(algebra::Expr) -> algebra::Expr + Clone + 'static,
          KernelInitializer: Fn(usize, usize) -> ndarray::Array2<f32>
{
    fn init(&self, namespace: &str) -> Box<LayerInstance> {
        let mut lv_builder = LayerVariablesBuilder::new(namespace);
        Box::new(DenseInstance{
            activation: self.activation.clone(),
            biases: lv_builder.append(ndarray::Array::zeros(self.output_size)),
            weights: lv_builder.append((self.kernel_initializer)(self.output_size, self.input_size)),
            variables: lv_builder.variables,
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
    where Activation: Fn(algebra::Expr) -> algebra::Expr
{
    activation: Activation,
    biases: algebra::Expr,
    weights: algebra::Expr,
    variables: Vec<super::LayerVariable>,
}

impl<Activation> LayerInstance for DenseInstance<Activation>
    where Activation: Fn(algebra::Expr) -> algebra::Expr
{
    fn expression(&self, input: algebra::Expr) -> algebra::Expr {
        (self.activation)(algebra::matvecmul(self.weights.clone(), input) + self.biases.clone())
    }

    fn variables(&self) -> &[super::LayerVariable] {
        self.variables.as_slice()
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
