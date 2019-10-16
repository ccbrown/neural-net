use super::super::{Layer, LayerInstance};

pub struct Reshape {
    pub shape: ndarray::IxDyn,
}

impl Layer for Reshape {
    fn init(&self, _namespace: &str, _input_shape: &ndarray::IxDyn) -> Box<LayerInstance> {
        let shape = self.shape.clone();
        Box::new(super::Instance{
            expression: move |input| input.reshape(shape.clone()),
            variables: vec![],
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use ndarray::Dimension;

    #[test]
    fn test() {
        let flat = ndarray::Array::range(0.0, 4.0, 1.0).into_dyn();
        let square = flat.clone().into_shape((2, 2)).unwrap().into_dyn();
        assert_eq!(Reshape{shape: ndarray::Ix2(2, 2).into_dyn()}.init("l", &flat.dim()).eval(flat.view()), square.into_dyn());
    }
}
