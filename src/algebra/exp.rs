use std::fmt;

use super::{Expr, ExprImpl};

#[derive(Clone)]
pub struct Exp {
    pub power: Expr,
}

impl ExprImpl for Exp {
    fn eval(&self) -> ndarray::ArrayD<f32> {
        self.power.eval().mapv(|v| v.exp())
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
            Expr::new(Self{
                power: self.power.propagate_constants(),
            })
        }
    }

    fn accumulate_gradients(&self, output: Expr, gradients: &mut super::Gradients) {
        self.power.accumulate_gradients(output.clone() * Expr::new(self.clone()), gradients);
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
        assert_eq!(format!("{}", (2.0 * x.clone()).exp().gradient("x")), "(exp((2 * x)) * 2)");

        let x = v("x", Rc::new(VariableValue::new(ndarray::arr1(&[0.0, 0.0, 0.0]))));
        assert_eq!(x.exp().gradient("x").eval(), ndarray::arr1(&[1.0, 1.0, 1.0]).into_dyn());
    }
}
