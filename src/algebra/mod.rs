use std::fmt;
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::atomic::{AtomicUsize, Ordering};

use ndarray::Dimension;

pub mod add; pub use add::*;
pub mod bias_add; pub use bias_add::*;
pub mod cmp; pub use cmp::*;
pub mod conv2d; pub use conv2d::*;
pub mod div; pub use div::*;
pub mod exp; pub use exp::*;
pub mod ternary; pub use ternary::*;
pub mod ln; pub use ln::*;
pub mod matmul; pub use matmul::*;
pub mod matvecmul; pub use matvecmul::*;
pub mod mul; pub use mul::*;
pub mod reshape; pub use reshape::*;
pub mod softmax; pub use softmax::*;
pub mod sub; pub use sub::*;
pub mod square; pub use square::*;
pub mod sum; pub use sum::*;
pub mod transpose; pub use transpose::*;
pub mod unevaluable; pub use unevaluable::*;
pub mod variable; pub use variable::*;
pub mod constant; pub use constant::*;

// All math done by the neural net is defined via Expr. This allows the library to perform
// automatic differentiation. This is the equivalent of a Tensor in Tensorflow.
//
// Some implementations such as softmax and square exist primarily for introspection. For example,
// if square didn't exist, we could just as easily write "x * x". However, `format!("{}",
// x.square())` is dramatically easier to read than `format!("{}", x * x)` when x is a complex
// expression. There are also potential performance benefits to having more specialized operations
// implemented.
#[derive(Clone)]
pub struct Expr {
    expr: Rc<ExprImpl>,
    id: usize,
}

pub struct Gradients {
    pub expressions: HashMap<String, Expr>,
}

pub trait ExprImpl: fmt::Display {
    fn eval_inputs(&self, _inputs: &Vec<ndarray::ArrayD<f32>>) -> ndarray::ArrayD<f32>;
    fn shape(&self) -> ndarray::IxDyn;
    fn is_constant(&self) -> bool;
    fn propagate_constants(&self) -> Expr;
    fn accumulate_gradients(&self, output: Expr, gradients: &mut Gradients);
    fn inputs(&self) -> Vec<&Expr>;

    fn eval(&self) -> ndarray::ArrayD<f32> {
        let mut inputs = Vec::new();
        for input in self.inputs() {
            inputs.push(input.eval());
        }
        self.eval_inputs(&inputs)
    }
}

static GLOBAL_EXPR_COUNT: AtomicUsize = AtomicUsize::new(0);

impl Expr {
    pub fn new<T: ExprImpl + 'static>(expr: T) -> Expr {
        Expr{
            expr: Rc::new(expr),
            id: GLOBAL_EXPR_COUNT.fetch_add(1, Ordering::SeqCst),
        }
    }

    // Returns a process-wide unique identifier for this expression and its clones.
    pub fn id(&self) -> usize {
        self.id
    }

    pub fn gradients(&self) -> HashMap<String, Expr> {
        let mut gradients = Gradients{
            expressions: HashMap::new(),
        };
        let output = expr(ndarray::Array::ones(self.shape()));
        self.accumulate_gradients(output, &mut gradients);
        gradients.expressions
    }

    pub fn gradient(&self, v: &str) -> Expr {
        self.gradients().get(v).unwrap_or(&expr(0.0)).simplified()
    }

    pub fn max(&self, b: Expr) -> Expr {
        ternary(
            cmp(self.clone(), cmp::Op::Less, b.clone()),
            b,
            self.clone(),
        )
    }

    pub fn softmax(&self) -> Expr {
        Expr::new(softmax::Softmax{
            expr: self.clone(),
        })
    }

    pub fn square(&self) -> Expr {
        Expr::new(square::Square{
            expr: self.clone(),
        })
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

    pub fn transpose(&self) -> Expr {
        Expr::new(transpose::Transpose{
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
    fn eval_inputs(&self, inputs: &Vec<ndarray::ArrayD<f32>>) -> ndarray::ArrayD<f32> {
        let result = self.expr.eval_inputs(inputs);
        if result.dim() != self.shape() {
            panic!("incorrect result shape for eval. got {:?}, expected {:?}", result.shape(), self.shape());
        }
        result
    }

    fn shape(&self) -> ndarray::IxDyn {
        self.expr.shape()
    }

    fn is_constant(&self) -> bool {
        self.expr.is_constant()
    }

    fn propagate_constants(&self) -> Expr {
        let result = self.expr.propagate_constants();
        if result.shape() != self.shape() {
            panic!("incorrect result shape for propagate_constants. got {:?}, expected {:?}", result.shape(), self.shape());
        }
        result
    }

    fn accumulate_gradients(&self, mut output: Expr, gradients: &mut Gradients) {
        if self.shape().ndim() == 0 && output.shape().ndim() > 0 {
            // reduce gradients to scalars when our ancestor broadcasts
            output = output.sum();
        }
        if output.shape() != self.shape() {
            panic!("incorrect output shape for accumulate_gradients. got {:?}, expected {:?}", output.shape(), self.shape());
        }
        self.expr.accumulate_gradients(output, gradients)
    }

    fn inputs(&self) -> Vec<&Expr> {
        self.expr.inputs()
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
