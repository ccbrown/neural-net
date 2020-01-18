use super::super::{Layer, LayerInstance};

use ndarray::Dimension;

pub struct Flatten {}

impl Layer for Flatten {
    fn init(
        self: Box<Self>,
        _namespace: &str,
        input_shape: &ndarray::IxDyn,
    ) -> Box<dyn LayerInstance> {
        let output_size = input_shape.size();
        Box::new(super::Instance {
            expression: move |input| input.reshape(ndarray::Ix1(output_size)),
            variables: vec![],
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test() {
        let flat = ndarray::Array::range(0.0, 4.0, 1.0);
        let square = flat.clone().into_shape((2, 2)).unwrap().into_dyn();
        assert_eq!(
            Box::new(Flatten {})
                .init("l", &square.dim())
                .eval(square.view()),
            flat.into_dyn()
        );
    }
}
