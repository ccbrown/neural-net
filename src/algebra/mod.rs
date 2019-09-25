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
    fn gradient(&self, v: &str) -> Expr;
    fn eval(&self) -> ndarray::ArrayD<f32>;
    fn shape(&self) -> ndarray::IxDyn;
    fn is_constant(&self) -> bool;
    fn propagate_constants(&self) -> Expr;
    fn freeze_dx(&self, v: &str, i: &ndarray::IxDyn) -> Expr;
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

    pub fn gradient_by_matrix(&self, v: &str, shape: ndarray::IxDyn) -> ndarray::ArrayD<Expr> {
        let du = self.expr.gradient(v).simplified();
        ndarray::Array::from_shape_fn(shape, |i| du.freeze_dx(v, &i))
    }

    pub fn gradient_by_scalar(&self, v: &str, i: &ndarray::IxDyn) -> Expr {
        self.expr.gradient(v).freeze_dx(v, &i).simplified()
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

    pub fn reshape<D: ndarray::Dimension>(&self, shape: D) -> Expr {
        Expr::new(reshape::Reshape{
            expr: self.clone(),
            shape: shape.into_dyn(),
        })
    }

    pub fn simplified(&self) -> Expr {
        self.propagate_constants()
    }
}

impl ExprImpl for Expr {
    fn gradient(&self, v: &str) -> Expr {
        self.expr.gradient(v).simplified()
    }

    fn eval(&self) -> ndarray::ArrayD<f32> {
        self.expr.eval()
    }

    fn shape(&self) -> ndarray::IxDyn {
        self.expr.shape()
    }

    fn is_constant(&self) -> bool {
        self.expr.is_constant()
    }

    fn propagate_constants(&self) -> Expr {
        self.expr.propagate_constants()
    }

    fn freeze_dx(&self, v: &str, i: &ndarray::IxDyn) -> Expr {
        self.expr.freeze_dx(v, i)
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
