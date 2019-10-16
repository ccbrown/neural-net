use super::super::{Layer, LayerInstance};

pub struct Sequential {
    pub layers: Vec<Box<Layer>>,
}

impl Layer for Sequential {
    fn init(&self, namespace: &str, input_shape: &ndarray::IxDyn) -> Box<LayerInstance> {
        let mut input_shape = input_shape.clone();
        let mut layers = Vec::new();
        let mut variables = Vec::new();
        for (i, layer) in self.layers.iter().enumerate() {
            let instance = layer.init(format!("{}/{}", namespace, i).as_str(), &input_shape);
            input_shape = instance.output_shape(&input_shape);
            for v in instance.variables() {
                variables.push(v.clone());
            }
            layers.push(instance);
        }
        Box::new(super::Instance{
            expression: move |input| {
                let mut result = input;
                for layer in layers.iter() {
                    result = layer.expression(result);
                }
                result
            },
            variables: variables,
        })
    }
}
