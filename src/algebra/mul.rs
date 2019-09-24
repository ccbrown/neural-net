use std::fmt;

use super::{Expr, ExprImpl};

// Mul performs element-wise multiplication.
pub struct Mul {
    pub left: Expr,
    pub right: Expr,
}

impl ExprImpl for Mul {
    fn gradient(&self, v: &str, i: &ndarray::IxDyn) -> Expr {
        // TODO: matrix-by-scalar
        self.left.clone() * self.right.gradient(v, i) + self.right.clone() * self.left.gradient(v, i)
    }

    fn eval(&self) -> ndarray::ArrayD<f32> {
        self.left.eval() * self.right.eval()
    }

    fn shape(&self) -> ndarray::IxDyn {
        self.left.shape()
    }
}

impl fmt::Display for Mul {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({} * {})", self.left, self.right)
    }
}

impl<T: Into<Expr>> std::ops::Mul<T> for Expr {
    type Output = Self;
    fn mul(self, rhs: T) -> Self {
        Expr::new(Mul{
            left: self,
            right: rhs.into(),
        })
    }
}

impl std::ops::Mul<Expr> for f32 {
    type Output = Expr;
    fn mul(self, rhs: Expr) -> Expr {
        Expr::new(Mul{
            left: super::expr(self),
            right: rhs,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::super::*;

    use ndarray::Dimension;

    #[test]
    fn test() {
        let x = v("x", Rc::new(VariableValue::new(ndarray::arr0(0.0))));
        assert_eq!(format!("{}", (3.0 * x).gradient("x", &ndarray::Ix0().into_dyn())), "((3 * 1) + (x * 0))");

        let x = v("x", Rc::new(VariableValue::new(ndarray::arr0(0.0))));
        let y = v("y", Rc::new(VariableValue::new(ndarray::arr0(0.0))));
        assert_eq!(format!("{}", (y * x).gradient("x", &ndarray::Ix0().into_dyn())), "((y * 1) + (x * 0))");
    }
}
