use super::super::{Layer, LayerInstance};

pub struct Residual {
    pub body: Box<dyn Layer>,
}

impl Layer for Residual {
    fn init(self: Box<Self>, namespace: &str, input_shape: &ndarray::IxDyn) -> Box<dyn LayerInstance> {
        let body = self.body.init(namespace, input_shape);
        let variables = body.variables().to_vec();
        Box::new(super::Instance{
            expression: move |input| body.expression(input.clone()) + input,
            variables: variables,
        })
    }
}
