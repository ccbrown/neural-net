use std::fmt;
use std::rc::Rc;

pub mod add; pub use add::*;
pub mod cmp; pub use cmp::*;
pub mod div; pub use div::*;
pub mod exp; pub use exp::*;
pub mod ternary; pub use ternary::*;
pub mod ln; pub use ln::*;
pub mod matvecmul; pub use matvecmul::*;
pub mod mul; pub use mul::*;
pub mod reshape; pub use reshape::*;
pub mod sub; pub use sub::*;
pub mod sum; pub use sum::*;
pub mod variable; pub use variable::*;
pub mod constant; pub use constant::*;

pub trait ExprImpl: fmt::Display {
    fn gradient(&self, v: &str, i: &ndarray::IxDyn) -> Expr;
    fn eval(&self) -> ndarray::ArrayD<f32>;
    fn shape(&self) -> ndarray::IxDyn;
}

#[derive(Clone)]
pub struct Expr {
    expr: Rc<ExprImpl>,
}

impl Expr {
    pub fn new<T: ExprImpl + 'static>(expr: T) -> Expr {
        Expr{
            expr: Rc::new(expr),
        }
    }

    pub fn exp(&self) -> Expr {
        Expr::new(exp::Exp{
            power: self.clone(),
        })
    }

    pub fn ln(&self) -> Expr {
        Expr::new(ln::Ln{
            expr: self.clone(),
        })
    }

    pub fn sum(&self) -> Expr {
        Expr::new(sum::Sum{
            expr: self.clone(),
        })
    }
}

impl ExprImpl for Expr {
    fn gradient(&self, v: &str, i: &ndarray::IxDyn) -> Expr {
        self.expr.gradient(v, i)
    }

    fn eval(&self) -> ndarray::ArrayD<f32> {
        self.expr.eval()
    }

    fn shape(&self) -> ndarray::IxDyn {
        self.expr.shape()
    }
}

impl std::ops::Deref for Expr {
    type Target = Rc<ExprImpl>;

    fn deref(&self) -> &Self::Target {
        &self.expr
    }
}

impl fmt::Display for Expr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.expr.fmt(f)
    }
}

impl fmt::Debug for Expr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.expr.fmt(f)
    }
}

pub fn expr<T: Into<Expr>>(e: T) -> Expr {
    e.into()
}
