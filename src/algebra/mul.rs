use std::fmt;

use ndarray::Dimension;

use super::{Expr, ExprImpl};

// Mul performs element-wise multiplication. If the numerator and denominator are not the same shape, one
// must be a scalar.
pub struct Mul {
    pub left: Expr,
    pub right: Expr,
}

impl ExprImpl for Mul {
    fn eval_inputs(&self, inputs: &Vec<ndarray::ArrayD<f32>>) -> ndarray::ArrayD<f32> {
        let (left, right) = (&inputs[0], &inputs[1]);
        // ndarray will broadcast a scalar into the larger operand, but only if it's on the right
        if self.left.shape().ndim() == 0 {
            right * left
        } else {
            left * right
        }
    }

    fn shape(&self) -> ndarray::IxDyn {
        let left = self.left.shape();
        if left.ndim() != 0 {
            left
        } else {
            self.right.shape()
        }
    }

    fn is_constant(&self) -> bool {
        let left = self.left.is_constant();
        let right = self.right.is_constant();
        if left && right {
            true
        } else if left {
            let left = self.left.eval();
            left == ndarray::Array::zeros(left.dim())
        } else if right {
            let right = self.right.eval();
            right == ndarray::Array::zeros(right.dim())
        } else {
            false
        }
    }

    fn propagate_constants(&self) -> Expr {
        if self.is_constant() {
            super::expr(self.eval())
        } else {
            if self.left.is_constant() {
                let left = self.left.eval();
                if left == ndarray::Array::ones(left.dim()) && self.shape() == self.right.shape() {
                    return self.right.propagate_constants();
                }
            }
            if self.right.is_constant() {
                let right = self.right.eval();
                if right == ndarray::Array::ones(right.dim()) && self.shape() == self.left.shape() {
                    return self.left.propagate_constants();
                }
            }
            self.left.propagate_constants() * self.right.propagate_constants()
        }
    }

    fn accumulate_gradients(&self, output: Expr, _gradients: &mut super::Gradients) -> Vec<Option<Expr>> {
        vec![Some(output.clone() * self.right.clone()), Some(output.clone() * self.left.clone())]
    }

    fn inputs(&self) -> Vec<&Expr> {
        vec![&self.left, &self.right]
    }
}

impl fmt::Display for Mul {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({} * {})", self.left, self.right)
    }
}

impl<T: Into<Expr>> std::ops::Mul<T> for Expr {
    type Output = Self;
    fn mul(self, rhs: T) -> Self {
        Expr::new(Mul{
            left: self,
            right: rhs.into(),
        })
    }
}

impl std::ops::Mul<Expr> for f32 {
    type Output = Expr;
    fn mul(self, rhs: Expr) -> Expr {
        Expr::new(Mul{
            left: super::expr(self),
            right: rhs,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::super::*;

    #[test]
    fn test() {
        let x = v("x", Rc::new(VariableValue::new(ndarray::arr0(0.0))));
        assert_eq!((3.0 * x.clone()).gradient("x").eval(), ndarray::arr0(3.0).into_dyn());

        let x = v("x", Rc::new(VariableValue::new(ndarray::arr0(0.0))));
        let y = v("y", Rc::new(VariableValue::new(ndarray::arr0(0.0))));
        assert_eq!(format!("{}", (y.clone() * x.clone()).gradient("x")), "y");

        let x = expr(ndarray::arr1(&[0.0, 1.0,  2.0]));
        let y = expr(ndarray::arr1(&[0.0, 1.0,  5.0]));
        let z = expr(ndarray::arr1(&[0.0, 1.0, 10.0]));
        assert_eq!((x * y).eval(), z.eval());

        let x = expr(2.0);
        let y = expr(ndarray::arr1(&[0.0, 1.0,  2.0]));
        let z = expr(ndarray::arr1(&[0.0, 2.0, 4.0]));
        assert_eq!((x.clone() * y.clone()).eval(), z.eval());
        assert_eq!((y * x).eval(), z.eval());

        let x = v("x", Rc::new(VariableValue::new(ndarray::arr0(2.0))));
        let y = expr(ndarray::arr1(&[1.0, 1.0, 1.0]));
        let z = expr(ndarray::arr1(&[2.0, 2.0, 2.0]));
        assert_eq!((x * y).propagate_constants().eval(), z.eval());

        let x = v("x", Rc::new(VariableValue::new(ndarray::arr1(&[0.0, 1.0, 2.0]))));
        let y = v("y", Rc::new(VariableValue::new(ndarray::arr1(&[0.0, 1.0, 5.0]))));
        assert_eq!((x.clone() * y).gradient("x").eval(), ndarray::arr1(&[0.0, 1.0, 5.0]).into_dyn());
    }
}
