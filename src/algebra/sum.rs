use std::fmt;

use ndarray::Dimension;

use super::{Expr, ExprImpl};

// Sums up every element in the expression into a scalar.
pub struct Sum {
    pub expr: Expr,
}

impl ExprImpl for Sum {
    fn eval_inputs(&self, inputs: &Vec<ndarray::ArrayD<f32>>) -> ndarray::ArrayD<f32> {
        ndarray::arr0(inputs[0].sum()).into_dyn()
    }

    fn shape(&self) -> ndarray::IxDyn {
        ndarray::Ix0().into_dyn()
    }

    fn is_constant(&self) -> bool {
        self.expr.is_constant()
    }

    fn propagate_constants(&self) -> Expr {
        if self.is_constant() {
            super::expr(self.eval())
        } else {
            self.expr.propagate_constants().sum()
        }
    }

    fn accumulate_gradients(&self, output: Expr, _gradients: &mut super::Gradients) -> Vec<Option<Expr>> {
        vec![Some(output * super::expr(ndarray::Array::ones(self.expr.shape())))]
    }

    fn inputs(&self) -> Vec<&Expr> {
        vec![&self.expr]
    }
}

impl fmt::Display for Sum {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "sum({})", self.expr)
    }
}

#[cfg(test)]
mod tests {
    use super::super::*;

    #[test]
    fn test() {
        let x = v("x", Rc::new(VariableValue::new(ndarray::arr1(&[0.0, 1.0, 2.0]))));
        assert_eq!(x.sum().gradient("x").eval(), ndarray::arr1(&[1.0, 1.0, 1.0]).into_dyn());

        let x = v("x", Rc::new(VariableValue::new(ndarray::arr1(&[0.0, 0.0, 0.0]))));
        assert_eq!(x.exp().sum().gradient("x").eval(), ndarray::arr1(&[1.0, 1.0, 1.0]).into_dyn());
    }
}
