use std::fmt;

use super::{Expr, ExprImpl};

pub struct Sqrt {
    pub expr: Expr,
}

// Computes the element-wise square root of the expression.
impl ExprImpl for Sqrt {
    fn eval_inputs(&self, inputs: &Vec<ndarray::ArrayD<f32>>) -> ndarray::ArrayD<f32> {
        inputs[0].mapv(|x| x.sqrt())
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
            self.expr.propagate_constants().sqrt()
        }
    }

    fn accumulate_gradients(&self, output: Expr, _gradients: &mut super::Gradients) -> Vec<Option<Expr>> {
        // TODO
        vec![Some(output.clone() * 2.0 * self.expr.clone())]
    }

    fn inputs(&self) -> Vec<&Expr> {
        vec![&self.expr]
    }
}

impl fmt::Display for Sqrt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "sqrt({})", self.expr)
    }
}
