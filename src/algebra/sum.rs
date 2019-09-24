use std::fmt;

use super::{Expr, ExprImpl};

use ndarray::Dimension;

// Sums up every element in the expression into a scalar.
pub struct Sum {
    pub expr: Expr,
}

impl ExprImpl for Sum {
    fn gradient(&self, v: &str, i: &ndarray::IxDyn) -> Expr {
        self.expr.gradient(v, i).sum()
    }

    fn eval(&self) -> ndarray::ArrayD<f32> {
        ndarray::arr0(self.expr.eval().sum()).into_dyn()
    }

    fn shape(&self) -> ndarray::IxDyn {
        ndarray::Ix0().into_dyn()
    }
}

impl fmt::Display for Sum {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "sum({})", self.expr)
    }
}
