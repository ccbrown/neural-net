use std::fmt;

use super::{c, Expr, ExprImpl};

pub struct Mul {
    pub left: Expr,
    pub right: Expr,
}

impl ExprImpl for Mul {
    fn gradient(&self, v: &str) -> Expr {
        self.left.clone() * self.right.gradient(v) + self.right.clone() * self.left.gradient(v)
    }

    fn eval(&self) -> f32 {
        self.left.eval() * self.right.eval()
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
            left: c(self),
            right: rhs,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::super::*;

    use std::cell::Cell;

    #[test]
    fn test() {
        let x = v("x", Rc::new(Cell::new(0.0)));
        assert_eq!(format!("{}", (3.0 * x).gradient("x")), "((3 * 1) + (x * 0))");

        let x = v("x", Rc::new(Cell::new(0.0)));
        let y = v("y", Rc::new(Cell::new(0.0)));
        assert_eq!(format!("{}", (y * x).gradient("x")), "((y * 1) + (x * 0))");
    }
}
