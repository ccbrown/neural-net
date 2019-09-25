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
    fn eval(&self) -> ndarray::ArrayD<f32>;
    fn shape(&self) -> ndarray::IxDyn;
    fn is_constant(&self) -> bool;
    fn propagate_constants(&self) -> Expr;

    // Replaces the variable with the given name with a constant.
    fn freeze_variable(&self, name: &str) -> Expr;

    // Calculates the gradient with respect to a given variable and returns an expression with an
    // "f'v" variable in it. The gradient can be completed for a given index within the variable by
    // setting or freezing the value of f'v and evaluating the expression.
    fn gradient(&self, v: &str, fv: &Rc<VariableValue>) -> Expr;

    fn variable_name(&self) -> Option<&str> {
        None
    }
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

    // Returns the gradient in respect to a given index within a variable.
    pub fn gradient_by_scalar(&self, v: &Expr, i: &ndarray::IxDyn) -> Expr {
        if let Some(name) = v.variable_name() {
            let mut dx = ndarray::Array::zeros(v.shape());
            dx[i] = 1.0;
            let dx = Rc::new(VariableValue::new(dx));
            let dx_name = "f'".to_string() + name;
            self.gradient(name, &dx).freeze_variable(&dx_name).simplified()
        } else {
            panic!("gradient_by_scalar argument must be a variable")
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
    fn gradient(&self, v: &str, fv: &Rc<VariableValue>) -> Expr {
        self.expr.gradient(v, fv)
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

    fn freeze_variable(&self, name: &str) -> Expr {
        self.expr.freeze_variable(name)
    }

    fn variable_name(&self) -> Option<&str> {
        self.expr.variable_name()
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
