use std::fmt;

use super::{Expr, ExprImpl};

pub struct Constant {
    pub value: f32,
}

impl ExprImpl for Constant {
    fn gradient(&self, _v: &str) -> Expr {
        Expr::new(Constant{
            value: 0.0,
        })
    }

    fn eval(&self) -> f32 {
        self.value
    }
}

impl fmt::Display for Constant {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.value)
    }
}

pub fn c(c: f32) -> Expr {
    Expr::new(Constant{
        value: c,
    })
}

impl From<f32> for Expr {
    fn from(f: f32) -> Expr {
        Expr::new(Constant{
            value: f,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::super::*;

    #[test]
    fn test() {
        assert_eq!(format!("{}", c(99.0).gradient("x")), "0");
    }
}
