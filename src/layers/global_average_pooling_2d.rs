use super::super::{algebra, Layer, LayerInstance};

pub struct GlobalAveragePooling2D {}

// GlobalAveragePooling2D takes a 3-dimensional input and produces a 1-dimensional output
// consisting of the averages of the elements in the first two dimensions.
impl Layer for GlobalAveragePooling2D {
    fn init(self: Box<Self>, _namespace: &str, input_shape: &ndarray::IxDyn) -> Box<dyn LayerInstance> {
        let input_shape = input_shape.clone();
        Box::new(super::Instance{
            expression: move |input| algebra::reduce_sum(input, vec![0, 1]).reshape(ndarray::Ix1(input_shape[2])) / (input_shape[0] * input_shape[1]) as f32,
            variables: vec![],
        })
    }
}
