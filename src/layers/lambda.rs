use super::super::{algebra, Layer, LayerInstance};

// Lambda simply applies a function to its input.
pub struct Lambda<F>
where
    F: Fn(algebra::Expr) -> algebra::Expr + 'static,
{
    pub f: F,
}

impl<F> Layer for Lambda<F>
where
    F: Fn(algebra::Expr) -> algebra::Expr + 'static,
{
    fn init(
        self: Box<Self>,
        _namespace: &str,
        _input_shape: &ndarray::IxDyn,
    ) -> Box<dyn LayerInstance> {
        Box::new(super::Instance {
            expression: self.f,
            variables: vec![],
        })
    }
}
