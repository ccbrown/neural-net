use std::fmt;

use ndarray::Dimension;

use super::{Expr, ExprImpl};

// Outputs one of two values based on a condition (1 or 0). If the true and false expressions are
// not the same shape, one must be a scalar. If the condition is not a scalar and the true and
// false expressions are not both scalars, the condition must be the same shape as the larger of
// the true and false expressions.
pub struct Ternary {
    pub condition: Expr,
    pub true_expr: Expr,
    pub false_expr: Expr,
}

fn expand(a: &ndarray::ArrayD<f32>, shape: ndarray::IxDyn) -> ndarray::ArrayD<f32> {
    if a.ndim() < shape.ndim() {
        ndarray::Array::from_elem(shape, *a.first().unwrap())
    } else {
        a.clone()
    }
}

impl ExprImpl for Ternary {
    fn eval_inputs(&self, inputs: &Vec<ndarray::ArrayD<f32>>) -> ndarray::ArrayD<f32> {
        let (condition, true_expr, false_expr) = (&inputs[0], &inputs[1], &inputs[2]);
        if condition.ndim() == 0 {
            if *condition.first().unwrap() == 0.0 {
                expand(false_expr, self.true_expr.shape())
            } else {
                expand(true_expr, self.false_expr.shape())
            }
        } else {
            let false_condition = 1.0 - condition;
            condition * true_expr + false_condition * false_expr
        }
    }

    fn shape(&self) -> ndarray::IxDyn {
        let true_shape = self.true_expr.shape();
        if true_shape.ndim() > 0 {
            true_shape
        } else {
            let false_shape = self.false_expr.shape();
            if false_shape.ndim() > 0 {
                false_shape
            } else {
                self.condition.shape()
            }
        }
    }

    fn is_constant(&self) -> bool {
        if !self.condition.is_constant() {
            return false;
        }
        let condition = self.condition.eval();
        let sum = condition.sum();
        if sum == condition.len() as f32 {
            self.true_expr.is_constant()
        } else if sum == 0.0 {
            self.false_expr.is_constant()
        } else {
            self.true_expr.is_constant() && self.false_expr.is_constant()
        }
    }

    fn propagate_constants(&self) -> Expr {
        if self.is_constant() {
            super::expr(self.eval())
        } else if self.condition.is_constant() {
            let condition = self.condition.propagate_constants();
            (condition.clone() * self.true_expr.clone() + (super::expr(1.0) - condition) * self.false_expr.clone()).propagate_constants()
        } else {
            Expr::new(Self{
                condition: self.condition.propagate_constants(),
                true_expr: self.true_expr.propagate_constants(),
                false_expr: self.false_expr.propagate_constants(),
            })
        }
    }

    fn accumulate_gradients(&self, output: Expr, gradients: &mut super::Gradients) {
        self.true_expr.accumulate_gradients(ternary(self.condition.clone(), output.clone(), super::expr(0.0)), gradients);
        self.false_expr.accumulate_gradients(ternary(self.condition.clone(), super::expr(0.0), output), gradients);
    }

    fn inputs(&self) -> Vec<&Expr> {
        vec![&self.condition, &self.true_expr, &self.false_expr]
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

#[cfg(test)]
mod tests {
    use super::super::*;

    #[test]
    fn test() {
        let c = expr(ndarray::arr1(&[0.0, 1.0]));
        let t = expr(ndarray::arr1(&[1.0, 2.0]));
        let f = expr(ndarray::arr0(3.0));
        let x = expr(ndarray::arr1(&[3.0, 2.0]));
        assert_eq!(ternary(c, t, f).eval(), x.eval());
    }
}
