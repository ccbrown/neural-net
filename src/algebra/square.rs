use std::fmt;

use super::{Expr, ExprImpl};

pub struct Square {
    pub expr: Expr,
}

impl ExprImpl for Square {
    fn eval(&self) -> ndarray::ArrayD<f32> {
        let v = self.expr.eval();
        &v * &v
    }

    fn shape(&self) -> ndarray::IxDyn {
        self.expr.shape()
    }

    fn is_constant(&self) -> bool {
        self.expr.is_constant()
    }

    fn propagate_constants(&self) -> Expr {
        if self.is_constant() {
            super::expr(self.eval())
        } else {
            self.expr.propagate_constants().square()
        }
    }

    fn accumulate_gradients(&self, output: Expr, gradients: &mut super::Gradients) {
        self.expr.accumulate_gradients(output.clone() * 2.0 * self.expr.clone(), gradients);
    }
}

impl fmt::Display for Square {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "square({})", self.expr)
    }
}
