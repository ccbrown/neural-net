use std::fmt;

use super::{Expr, ExprImpl};

pub struct Add {
    pub left: Expr,
    pub right: Expr,
}

impl ExprImpl for Add {
    fn gradient(&self, v: &str) -> Expr {
        self.left.gradient(v) + self.right.gradient(v)
    }

    fn eval(&self) -> f32 {
        self.left.eval() + self.right.eval()
    }
}

impl fmt::Display for Add {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({} + {})", self.left, self.right)
    }
}

impl<T: Into<Expr>> std::ops::Add<T> for Expr {
    type Output = Self;
    fn add(self, rhs: T) -> Self {
        Expr::new(Add{
            left: self,
            right: rhs.into(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::super::*;

    #[test]
    fn test() {
        let x = v("x", Rc::new(0.0));
        let y = v("y", Rc::new(0.0));
        assert_eq!(format!("{}", (x + y).gradient("x")), "(1 + 0)");

        let x = v("x", Rc::new(0.0));
        let y = v("y", Rc::new(0.0));
        assert_eq!(format!("{}", (x + y).gradient("y")), "(0 + 1)");
    }
}
