use std::fmt;

use super::{Expr, ExprImpl};

// Outputs one of two values based on a condition (1 or 0).
pub struct Ternary {
    pub condition: Expr,
    pub true_expr: Expr,
    pub false_expr: Expr,
}

impl ExprImpl for Ternary {
    fn gradient(&self, v: &str) -> Expr {
        Expr::new(Ternary{
            condition: self.condition.clone(),
            true_expr: self.true_expr.gradient(v),
            false_expr: self.false_expr.gradient(v),
        })
    }

    fn eval(&self) -> f32 {
        if self.condition.eval() == 0.0 {
            self.false_expr.eval()
        } else {
            self.true_expr.eval()
        }
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
