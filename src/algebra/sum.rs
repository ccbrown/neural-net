use std::fmt;

use super::{Expr, ExprImpl};

use ndarray::Dimension;

// Sums up every element in the expression into a scalar.
pub struct Sum {
    pub expr: Expr,
}

impl ExprImpl for Sum {
    fn gradient(&self, v: &str) -> Expr {
        self.expr.gradient(v).sum()
    }

    fn eval(&self) -> ndarray::ArrayD<f32> {
        ndarray::arr0(self.expr.eval().sum()).into_dyn()
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

impl fmt::Display for Sum {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "sum({})", self.expr)
    }
}

#[cfg(test)]
mod tests {
    use super::super::*;

    use ndarray::Dimension;

    #[test]
    fn test() {
        let x = v("x", Rc::new(VariableValue::new(ndarray::arr2(&[[0.0, 1.0], [2.0, 3.0]]))));
        assert_eq!(x.sum().gradient_by_scalar("x", &ndarray::Ix2(1, 0).into_dyn()).eval(), ndarray::arr0(1.0).into_dyn());
    }
}
