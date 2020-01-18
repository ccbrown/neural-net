use std::fmt;

use super::{Expr, ExprImpl};

#[derive(Clone)]
pub struct Exp {
    pub power: Expr,
}

impl ExprImpl for Exp {
    fn eval_inputs(&self, inputs: &Vec<ndarray::ArrayD<f32>>) -> ndarray::ArrayD<f32> {
        inputs[0].mapv(|v| v.exp())
    }

    fn shape(&self) -> ndarray::IxDyn {
        self.power.shape()
    }

    fn is_constant(&self) -> bool {
        self.power.is_constant()
    }

    fn propagate_constants(&self) -> Expr {
        if self.is_constant() {
            super::expr(self.eval())
        } else {
            self.power.propagate_constants().exp()
        }
    }

    fn accumulate_gradients(
        &self,
        output: Expr,
        _gradients: &mut super::Gradients,
    ) -> Vec<Option<Expr>> {
        vec![Some(output.clone() * Expr::new(self.clone()))]
    }

    fn inputs(&self) -> Vec<&Expr> {
        vec![&self.power]
    }
}

impl fmt::Display for Exp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "exp({})", self.power)
    }
}

#[cfg(test)]
mod tests {
    use super::super::*;

    #[test]
    fn test() {
        let x = v("x", Rc::new(VariableValue::new(ndarray::arr0(0.0))));
        assert_eq!(
            format!("{}", (2.0 * x.clone()).exp().gradient("x")),
            "(exp((2 * x)) * 2)"
        );

        let x = v(
            "x",
            Rc::new(VariableValue::new(ndarray::arr1(&[0.0, 0.0, 0.0]))),
        );
        assert_eq!(
            x.exp().gradient("x").eval(),
            ndarray::arr1(&[1.0, 1.0, 1.0]).into_dyn()
        );
    }
}
