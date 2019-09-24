use std::rc::Rc;

use super::{algebra, Layer, LayerInstance, Shape};

pub struct Flatten {
    pub input_shape: Shape,
}

impl Layer for Flatten {
    type Instance = FlattenInstance;

    fn init(&self, _namespace: &str) -> Self::Instance {
        FlattenInstance{
            output_size: self.input_shape.size(),
        }
    }

    fn input_shape(&self) -> Shape {
        self.input_shape
    }

    fn output_shape(&self) -> Shape {
        Shape::D1(self.input_shape.size())
    }
}

pub struct FlattenInstance {
    output_size: usize,
}

impl LayerInstance for FlattenInstance {
    fn expression<'a, D: ndarray::Dimension>(&self, input: ndarray::ArrayView<'a, algebra::Expr, D>) -> ndarray::ArrayD<algebra::Expr> {
        input.into_shape(ndarray::Ix1(self.output_size)).unwrap().to_owned().into_dyn()
    }
}

// Dense takes a 1-dimensional input and outputs a 1-dimensional output.
pub struct Dense<Activation, KernelInitializer>
    where Activation: Fn(ndarray::ArrayD<algebra::Expr>) -> ndarray::ArrayD<algebra::Expr> + Clone,
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
            value: Rc::new(init),
        };
        self.trainable_variables.push(tv.clone());
        algebra::v(tv.name, tv.value)
    }
}

impl<Activation, KernelInitializer> Layer for Dense<Activation, KernelInitializer>
    where Activation: Fn(ndarray::ArrayD<algebra::Expr>) -> ndarray::ArrayD<algebra::Expr> + Clone,
          KernelInitializer: Fn(usize, usize) -> ndarray::Array2<f32>
{
    type Instance = DenseInstance<Activation>;

    fn init(&self, namespace: &str) -> Self::Instance {
        let mut tv_builder = TrainableVariablesBuilder::new(namespace);
        DenseInstance{
            activation: self.activation.clone(),
            biases: ndarray::Array::zeros(self.output_size).mapv(|x| tv_builder.append(x)),
            weights: (self.kernel_initializer)(self.output_size, self.input_size).mapv(|x| tv_builder.append(x)),
            trainable_variables: tv_builder.trainable_variables,
        }
    }

    fn input_shape(&self) -> Shape {
        Shape::D1(self.input_size)
    }

    fn output_shape(&self) -> Shape {
        Shape::D1(self.output_size)
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
    fn expression<'a, D: ndarray::Dimension>(&self, input: ndarray::ArrayView<'a, algebra::Expr, D>) -> ndarray::ArrayD<algebra::Expr> {
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
            input_shape: Shape::D2(2, 2),
        }.init("l").eval(square.view()), flat.into_dyn());
    }

    #[test]
    fn test_dense() {
        let a = ndarray::Array::range(0.0, 16.0, 1.0);
        assert_eq!(Dense{
            activation: |x| x + 1.0,
            kernel_initializer: super::super::initializers::zeros,
            input_size: 16,
            output_size: 4,
        }.init("l").eval(a.view()), ndarray::Array::ones(4).into_dyn());
    }
}
