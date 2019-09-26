use std::fmt;

use super::{Expr, ExprImpl};

pub struct Softmax {
    pub expr: Expr,
}

impl ExprImpl for Softmax {
    fn eval(&self) -> ndarray::ArrayD<f32> {
        let v = self.expr.exp().eval();
        let sum = v.sum();
        v / sum
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
            return Expr::new(Self{
                expr: self.expr.propagate_constants(),
            });
        }
    }

    fn accumulate_gradients(&self, output: Expr, gradients: &mut super::Gradients) {
        let softmax = softmax(self.expr.clone());
        self.expr.accumulate_gradients((output.clone() - (output.clone() * softmax.clone()).sum()) * softmax, gradients);
    }
}

impl fmt::Display for Softmax {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "softmax({})", self.expr)
    }
}

pub fn softmax(expr: Expr) -> Expr {
    Expr::new(Softmax{
        expr: expr,
    })
}
