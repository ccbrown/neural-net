use std::fmt;

use super::{Expr, ExprImpl};

pub struct Ln {
    pub expr: Expr,
}

impl ExprImpl for Ln {
    fn eval(&self) -> ndarray::ArrayD<f32> {
        self.expr.eval().mapv(|v| v.ln())
    }

    fn shape(&self) -> ndarray::IxDyn {
        self.expr.shape()
    }

    fn is_constant(&self) -> bool {
        self.expr.is_constant()
    }

    fn propagate_constants(&self) -> Expr {
        if self.is_constant() {
            super::expr(self.eval())
        } else {
            self.expr.propagate_constants().ln()
        }
    }

    fn accumulate_gradients(&self, output: Expr, gradients: &mut super::Gradients) {
        self.expr.accumulate_gradients(output.clone() / self.expr.clone(), gradients);
    }
}

impl fmt::Display for Ln {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ln({})", self.expr)
    }
}

#[cfg(test)]
mod tests {
    use super::super::*;

    #[test]
    fn test() {
        let x = v("x", Rc::new(VariableValue::new(ndarray::arr0(0.0))));
        assert_eq!(format!("{}", (2.0 * x.clone()).ln().gradient("x")), "((1 / (2 * x)) * 2)");
    }
}
