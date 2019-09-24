use std::fmt;
use std::rc::Rc;

pub mod add; pub use add::*;
pub mod cmp; pub use cmp::*;
pub mod div; pub use div::*;
pub mod exp; pub use exp::*;
pub mod ternary; pub use ternary::*;
pub mod mul; pub use mul::*;
pub mod sub; pub use sub::*;
pub mod variable; pub use variable::*;
pub mod constant; pub use constant::*;

pub trait ExprImpl: fmt::Display {
    fn gradient(&self, v: &str) -> Expr;
    fn eval(&self) -> f32;
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
}

impl ExprImpl for Expr {
    fn gradient(&self, v: &str) -> Expr {
        self.expr.gradient(v)
    }

    fn eval(&self) -> f32 {
        self.expr.eval()
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

pub fn vec_dot<S1, S2>(a: &ndarray::ArrayBase<S1, ndarray::Ix1>, b: &ndarray::ArrayBase<S2, ndarray::Ix1>) -> Expr
    where S1: ndarray::Data<Elem=Expr>,
          S2: ndarray::Data<Elem=Expr>,
{
    let mut result = ndarray::Array::from_elem(a.len(), c(0.0));
    ndarray::Zip::from(&mut result)
        .and(a)
        .and(b)
        .apply(|out, a, b| *out = a.clone() * b.clone());
    result.fold(c(0.0), |sum, e| sum + e.clone())
}

pub fn mat_vec_mul<S1, S2>(a: &ndarray::ArrayBase<S1, ndarray::Ix2>, x: &ndarray::ArrayBase<S2, ndarray::Ix1>) -> ndarray::Array1<Expr>
    where S1: ndarray::Data<Elem=Expr>,
          S2: ndarray::Data<Elem=Expr>,
{
    let mut result = ndarray::Array::from_elem(a.rows(), c(0.0));
    ndarray::Zip::from(a.outer_iter())
        .and(&mut result)
        .apply(|row, out| *out = vec_dot(&row, &x));
    result
}

