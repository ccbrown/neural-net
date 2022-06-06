use std::rc::Rc;

use super::{algebra, LayerInstance, LayerVariable};

pub mod batch_normalization;
pub use batch_normalization::*;
pub mod conv2d;
pub use conv2d::*;
pub mod dense;
pub use dense::*;
pub mod flatten;
pub use flatten::*;
pub mod global_average_pooling_2d;
pub use global_average_pooling_2d::*;
pub mod lambda;
pub use lambda::*;
pub mod residual;
pub use residual::*;
pub mod sequential;
pub use sequential::*;

struct LayerVariablesBuilder {
    namespace: String,
    variables: Vec<super::LayerVariable>,
}

impl LayerVariablesBuilder {
    fn new<S: Into<String>>(namespace: S) -> Self {
        LayerVariablesBuilder {
            namespace: namespace.into(),
            variables: Vec::new(),
        }
    }

    fn append<S1, D>(&mut self, name: &str, init: ndarray::ArrayBase<S1, D>) -> algebra::Expr
    where
        S1: ndarray::Data<Elem = f32>,
        D: ndarray::Dimension,
    {
        let v = super::LayerVariable {
            name: format!("{}.{}", self.namespace, name),
            value: Rc::new(algebra::VariableValue::new(init)),
        };
        self.variables.push(v.clone());
        algebra::v(v.name, v.value)
    }
}

pub struct Instance<F>
where
    F: Fn(algebra::Expr) -> algebra::Expr,
{
    expression: F,
    variables: Vec<LayerVariable>,
}

impl<F> LayerInstance for Instance<F>
where
    F: Fn(algebra::Expr) -> algebra::Expr,
{
    fn expression(&self, input: algebra::Expr) -> algebra::Expr {
        (self.expression)(input)
    }

    fn variables(&self) -> &[LayerVariable] {
        self.variables.as_slice()
    }
}
