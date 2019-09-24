use std::fmt;

use super::{Expr, ExprImpl};

pub struct Exp {
    pub power: Expr,
}

impl ExprImpl for Exp {
    fn gradient(&self, _v: &str) -> Expr {
        Expr::new(Exp{
            power: self.power.clone(),
        })
    }

    fn eval(&self) -> f32 {
        self.power.eval().exp()
    }
}

impl fmt::Display for Exp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "(e^{})", self.power)
    }
}
