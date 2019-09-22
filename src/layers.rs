use std::error::Error;

use super::{Layer, LayerInstance, Shape};

pub struct Flatten {
    pub input_shape: Shape,
}

impl Layer for Flatten {
    type Instance = FlattenInstance;

    fn init(&self) -> Self::Instance {
        FlattenInstance{}
    }

    fn input_shape(&self) -> Shape {
        self.input_shape
    }

    fn output_shape(&self) -> Shape {
        Shape::D1(self.input_shape.size())
    }
}

pub struct FlattenInstance {}

impl LayerInstance for FlattenInstance {
    fn eval<'a, D>(&self, input: ndarray::ArrayView<'a, f32, D>) -> Result<ndarray::ArrayD<f32>, Box<Error>>
        where D: ndarray::Dimension
    {
        let shape = ndarray::Ix1(input.len());
        Ok(input.into_shape(shape)?.into_owned().into_dyn())
    }
}

pub struct Dense<Activation, KernelInitializer>
    where Activation: Fn(ndarray::ArrayD<f32>) -> ndarray::ArrayD<f32> + Clone,
          KernelInitializer: Fn(usize, usize) -> ndarray::Array2<f32>
{
    pub activation: Activation,
    pub kernel_initializer: KernelInitializer,
    pub input_size: usize,
    pub output_size: usize,
}

impl<Activation, KernelInitializer> Layer for Dense<Activation, KernelInitializer>
    where Activation: Fn(ndarray::ArrayD<f32>) -> ndarray::ArrayD<f32> + Clone,
          KernelInitializer: Fn(usize, usize) -> ndarray::Array2<f32>
{
    type Instance = DenseInstance<Activation>;

    fn init(&self) -> Self::Instance {
        DenseInstance{
            activation: self.activation.clone(),
            biases: ndarray::Array::zeros(self.output_size),
            weights: (self.kernel_initializer)(self.output_size, self.input_size),
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
    where Activation: Fn(ndarray::ArrayD<f32>) -> ndarray::ArrayD<f32>
{
    activation: Activation,
    biases: ndarray::Array1<f32>,
    weights: ndarray::Array2<f32>,
}

impl<Activation> LayerInstance for DenseInstance<Activation>
    where Activation: Fn(ndarray::ArrayD<f32>) -> ndarray::ArrayD<f32>
{
    fn eval<'a, D>(&self, input: ndarray::ArrayView<'a, f32, D>) -> Result<ndarray::ArrayD<f32>, Box<Error>>
        where D: ndarray::Dimension
    {
        let input = input.into_dimensionality::<ndarray::Ix1>()?;
        let output = ndarray::Array::from_iter((0..self.biases.len()).map(|i| {
            self.weights.row(i).dot(&input) + self.biases[i]
        })).into_dyn();
        Ok((self.activation)(output))
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
        }.init().eval(square.view()).unwrap(), flat.into_dyn());
    }

    #[test]
    fn test_dense() {
        let a = ndarray::Array::range(0.0, 16.0, 1.0);
        assert_eq!(Dense{
            activation: |x| x + 1.0,
            kernel_initializer: super::super::initializers::zeros,
            input_size: 16,
            output_size: 4,
        }.init().eval(a.view()).unwrap(), ndarray::Array::ones(4).into_dyn());
    }
}
