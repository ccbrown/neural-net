use std::fmt;

use ndarray::Dimension;

use super::{Expr, ExprImpl};

pub struct Transpose {
    pub expr: Expr,
}

impl ExprImpl for Transpose {
    fn eval(&self) -> ndarray::ArrayD<f32> {
        self.expr.eval().t().into_owned()
    }

    fn shape(&self) -> ndarray::IxDyn {
        let mut shape = self.expr.shape().clone();
        shape.slice_mut().reverse();
        shape
    }

    fn is_constant(&self) -> bool {
        self.expr.is_constant()
    }

    fn propagate_constants(&self) -> Expr {
        if self.is_constant() {
            super::expr(self.eval())
        } else {
            Expr::new(Self{
                expr: self.expr.propagate_constants(),
            })
        }
    }

    fn accumulate_gradients(&self, output: Expr, gradients: &mut super::Gradients) {
        self.expr.accumulate_gradients(output.transpose(), gradients);
    }
}

impl fmt::Display for Transpose {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "transpose({})", self.expr)
    }
}
