use std::fmt;

use ndarray::Dimension;

use super::{Expr, ExprImpl};

pub struct Transpose {
    pub expr: Expr,
}

impl ExprImpl for Transpose {
    fn eval_inputs(&self, inputs: &Vec<ndarray::ArrayD<f32>>) -> ndarray::ArrayD<f32> {
        inputs[0].t().into_owned()
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
            self.expr.propagate_constants().transpose()
        }
    }

    fn accumulate_gradients(
        &self,
        output: Expr,
        _gradients: &mut super::Gradients,
    ) -> Vec<Option<Expr>> {
        vec![Some(output.transpose())]
    }

    fn inputs(&self) -> Vec<&Expr> {
        vec![&self.expr]
    }
}

impl fmt::Display for Transpose {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "transpose({})", self.expr)
    }
}
