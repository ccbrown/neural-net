use std::fmt;

use super::{Expr, ExprImpl};

pub struct Softmax {
    pub expr: Expr,
}

impl ExprImpl for Softmax {
    fn eval_inputs(&self, inputs: &Vec<ndarray::ArrayD<f32>>) -> ndarray::ArrayD<f32> {
        let v = inputs[0].mapv(|v| v.exp());
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
            self.expr.propagate_constants().softmax()
        }
    }

    fn accumulate_gradients(&self, output: Expr, _gradients: &mut super::Gradients) -> Vec<Option<Expr>> {
        let softmax = softmax(self.expr.clone());
        vec![Some((output.clone() - (output.clone() * softmax.clone()).sum()) * softmax)]
    }

    fn inputs(&self) -> Vec<&Expr> {
        vec![&self.expr]
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
