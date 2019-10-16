use std::fmt;

use super::{Expr, ExprImpl};

// Unevaluable represents an expression for which it is an error to attempt to evaluate.
pub struct Unevaluable {
    pub reason: String,
    pub shape: ndarray::IxDyn,
}

impl ExprImpl for Unevaluable {
    fn eval_inputs(&self, _inputs: &Vec<ndarray::ArrayD<f32>>) -> ndarray::ArrayD<f32> {
        panic!("{}", self.reason)
    }

    fn shape(&self) -> ndarray::IxDyn {
        self.shape.clone()
    }

    fn is_constant(&self) -> bool {
        false
    }

    fn propagate_constants(&self) -> Expr {
        unevaluable(self.shape.clone(), self.reason.clone())
    }

    fn accumulate_gradients(&self, _output: Expr, _gradients: &mut super::Gradients) {}

    fn inputs(&self) -> Vec<&Expr> {
        vec![]
    }
}

impl fmt::Display for Unevaluable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "unevaluable({})", self.reason)
    }
}

pub fn unevaluable<R: Into<String>>(shape: ndarray::IxDyn, reason: R) -> Expr {
    Expr::new(Unevaluable{
        reason: reason.into(),
        shape: shape,
    })
}
