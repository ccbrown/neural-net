use std::fmt;

use super::{Expr, ExprImpl};

pub struct Ln {
    pub expr: Expr,
}

impl ExprImpl for Ln {
    fn gradient(&self, v: &str) -> Expr {
        self.expr.gradient(v) / self.expr.clone()
    }

    fn eval(&self) -> ndarray::ArrayD<f32> {
        self.expr.eval().mapv(|v| v.ln())
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
            Expr::new(Self{
                expr: self.expr.propagate_constants(),
            })
        }
    }

    fn freeze_dx(&self, v: &str, i: &ndarray::IxDyn) -> Expr {
        Expr::new(Self{
            expr: self.expr.freeze_dx(v, i),
        })
    }
}

impl fmt::Display for Ln {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ln({})", self.expr)
    }
}

#[cfg(test)]
mod tests {
    use super::super::*;

    use ndarray::Dimension;

    #[test]
    fn test() {
        let x = v("x", Rc::new(VariableValue::new(ndarray::arr0(0.0))));
        assert_eq!(format!("{}", (2.0 * x).ln().gradient_by_scalar("x", &ndarray::Ix0().into_dyn())), "(2 / (2 * x))");
    }
}
