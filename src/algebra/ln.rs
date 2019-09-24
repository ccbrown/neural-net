use std::fmt;

use super::{Expr, ExprImpl};

pub struct Ln {
    pub expr: Expr,
}

impl ExprImpl for Ln {
    fn gradient(&self, v: &str, i: &ndarray::IxDyn) -> Expr {
        // TODO: matrix-by-scalar
        self.expr.gradient(v, i) / self.expr.clone()
    }

    fn eval(&self) -> ndarray::ArrayD<f32> {
        self.expr.eval().mapv(|v| v.ln())
    }

    fn shape(&self) -> ndarray::IxDyn {
        self.expr.shape()
    }
}

impl fmt::Display for Ln {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ln({})", self.expr)
    }
}
