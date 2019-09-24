use std::fmt;

use super::{Expr, ExprImpl};

pub struct Reshape {
    pub expr: Expr,
    pub shape: ndarray::IxDyn,
}

impl ExprImpl for Reshape {
    fn gradient(&self, v: &str, i: &ndarray::IxDyn) -> Expr {
        reshape(self.expr.gradient(v, i), self.shape.clone())
    }

    fn eval(&self) -> ndarray::ArrayD<f32> {
        self.expr.eval().into_shape(self.shape.clone()).unwrap()
    }

    fn shape(&self) -> ndarray::IxDyn {
        self.shape.clone()
    }
}

impl fmt::Display for Reshape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "reshape({})", self.expr)
    }
}

pub fn reshape<A: Into<Expr>>(a: A, shape: ndarray::IxDyn) -> Expr {
    Expr::new(Reshape{
        expr: a.into(),
        shape: shape,
    })
}
