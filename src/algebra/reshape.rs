use std::fmt;

use super::{Expr, ExprImpl};

pub struct Reshape {
    pub expr: Expr,
    pub shape: ndarray::IxDyn,
}

impl ExprImpl for Reshape {
    fn eval_inputs(&self, inputs: &Vec<ndarray::ArrayD<f32>>) -> ndarray::ArrayD<f32> {
        inputs[0].clone().into_shape(self.shape.clone()).unwrap()
    }

    fn shape(&self) -> ndarray::IxDyn {
        self.shape.clone()
    }

    fn is_constant(&self) -> bool {
        self.expr.is_constant()
    }

    fn propagate_constants(&self) -> Expr {
        if self.is_constant() {
            super::expr(self.eval())
        } else {
            self.expr.propagate_constants().reshape(self.shape.clone())
        }
    }

    fn accumulate_gradients(&self, output: Expr, gradients: &mut super::Gradients) {
        self.expr.accumulate_gradients(output.reshape(self.expr.shape()), gradients);
    }

    fn inputs(&self) -> Vec<&Expr> {
        vec![&self.expr]
    }
}

impl fmt::Display for Reshape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "reshape({})", self.expr)
    }
}

#[cfg(test)]
mod tests {
    use super::super::*;

    #[test]
    fn test() {
        let x = v("x", Rc::new(VariableValue::new(ndarray::arr2(&[[0.0, 1.0], [2.0, 3.0]]))));
        assert_eq!(x.reshape(ndarray::Ix1(4)).gradient("x").eval(), ndarray::arr2(&[[1.0, 1.0], [1.0, 1.0]]).into_dyn());
    }
}
