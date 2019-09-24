use std::fmt;

use super::{c, Expr, ExprImpl};

pub struct Div {
    pub num: Expr,
    pub den: Expr,
}

impl ExprImpl for Div {
    fn gradient(&self, v: &str) -> Expr {
        let num = self.num.gradient(v) * self.den.clone() - self.den.gradient(v) * self.num.clone();
        let den = self.den.clone() * self.den.clone();
        num / den
    }

    fn eval(&self) -> f32 {
        self.num.eval() / self.den.eval()
    }
}

impl fmt::Display for Div {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({} / {})", self.num, self.den)
    }
}

impl<T: Into<Expr>> std::ops::Div<T> for Expr {
    type Output = Self;
    fn div(self, rhs: T) -> Self {
        Expr::new(Div{
            num: self,
            den: rhs.into(),
        })
    }
}

impl std::ops::Div<Expr> for f32 {
    type Output = Expr;
    fn div(self, rhs: Expr) -> Expr {
        Expr::new(Div{
            num: c(self),
            den: rhs,
        })
    }
}
