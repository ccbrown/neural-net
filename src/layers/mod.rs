use std::rc::Rc;

use super::{algebra};

pub mod conv2d; pub use conv2d::*;
pub mod dense; pub use dense::*;
pub mod flatten; pub use flatten::*;

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

    fn append<S1, D>(&mut self, name: &str, init: ndarray::ArrayBase<S1, D>) -> algebra::Expr
        where S1: ndarray::Data<Elem=f32>,
              D: ndarray::Dimension,
    {
        let v = super::LayerVariable{
            name: format!("{}.{}", self.namespace, name),
            value: Rc::new(algebra::VariableValue::new(init)),
        };
        self.variables.push(v.clone());
        algebra::v(v.name, v.value)
    }
}
