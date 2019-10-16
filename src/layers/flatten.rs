use super::super::{algebra, Layer, LayerInstance};

use ndarray::Dimension;

pub struct Flatten {}

impl Layer for Flatten {
    fn init(&self, _namespace: &str, input_shape: &ndarray::IxDyn) -> Box<LayerInstance> {
        Box::new(FlattenInstance{
            output_size: input_shape.size(),
        })
    }

    fn output_shape(&self, input_shape: &ndarray::IxDyn) -> ndarray::IxDyn {
        ndarray::Ix1(input_shape.size()).into_dyn()
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test() {
        let flat = ndarray::Array::range(0.0, 4.0, 1.0);
        let square = flat.clone().into_shape((2, 2)).unwrap().into_dyn();
        assert_eq!(Flatten{}.init("l", &square.dim()).eval(square.view()), flat.into_dyn());
    }
}
