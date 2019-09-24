use std::fmt;

use super::{Expr, ExprImpl};

// Outputs one of two values based on a condition (1 or 0). The true and false expressions must be
// the same shape. If the condition is not a scalar, it must also be the same shape.
pub struct Ternary {
    pub condition: Expr,
    pub true_expr: Expr,
    pub false_expr: Expr,
}

impl ExprImpl for Ternary {
    fn gradient(&self, v: &str, i: &ndarray::IxDyn) -> Expr {
        Expr::new(Ternary{
            condition: self.condition.clone(),
            true_expr: self.true_expr.gradient(v, i),
            false_expr: self.false_expr.gradient(v, i),
        })
    }

    fn eval(&self) -> ndarray::ArrayD<f32> {
        let mut condition = self.condition.eval();
        if condition.ndim() == 0 {
            if *condition.first().unwrap() == 0.0 {
                self.false_expr.eval()
            } else {
                self.true_expr.eval()
            }
        } else {
            let false_expr = self.false_expr.eval();
            let true_expr = self.true_expr.eval();
            for (i, v) in condition.indexed_iter_mut() {
                *v = if *v == 0.0 { false_expr[i] } else { true_expr[i] };
            }
            condition
        }
    }

    fn shape(&self) -> ndarray::IxDyn {
        self.true_expr.shape()
    }
}

impl fmt::Display for Ternary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({} ? {} : {})", self.condition, self.true_expr, self.false_expr)
    }
}

pub fn ternary(condition: Expr, true_expr: Expr, false_expr: Expr) -> Expr {
    Expr::new(Ternary{
        condition: condition,
        true_expr: true_expr,
        false_expr: false_expr,
    })
}
