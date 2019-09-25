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
    fn gradient(&self, v: &str) -> Expr {
        self.left.clone() * self.right.gradient(v) + self.right.clone() * self.left.gradient(v)
    }

    fn eval(&self) -> ndarray::ArrayD<f32> {
        // ndarray will broadcast a scalar into the larger operand, but only if it's on the right
        if self.left.shape().ndim() == 0 {
            self.right.eval() * self.left.eval()
        } else {
            self.left.eval() * self.right.eval()
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
                if left == ndarray::Array::ones(left.dim()) {
                    return self.right.propagate_constants();
                }
            }
            if self.right.is_constant() {
                let right = self.right.eval();
                if right == ndarray::Array::ones(right.dim()) {
                    return self.left.propagate_constants();
                }
            }
            return Expr::new(Self{
                left: self.left.propagate_constants(),
                right: self.right.propagate_constants(),
            });
        }
    }

    fn freeze_dx(&self, v: &str, i: &ndarray::IxDyn) -> Expr {
        Expr::new(Self{
            left: self.left.freeze_dx(v, i),
            right: self.right.freeze_dx(v, i),
        })
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

    use ndarray::Dimension;

    #[test]
    fn test() {
        let x = v("x", Rc::new(VariableValue::new(ndarray::arr0(0.0))));
        assert_eq!((3.0 * x).gradient_by_scalar("x", &ndarray::Ix0().into_dyn()).eval(), ndarray::arr0(3.0).into_dyn());

        let x = v("x", Rc::new(VariableValue::new(ndarray::arr0(0.0))));
        let y = v("y", Rc::new(VariableValue::new(ndarray::arr0(0.0))));
        assert_eq!((y.clone() * x).gradient_by_scalar("x", &ndarray::Ix0().into_dyn()).eval(), y.eval());

        let x = expr(ndarray::arr1(&[0.0, 1.0,  2.0]));
        let y = expr(ndarray::arr1(&[0.0, 1.0,  5.0]));
        let z = expr(ndarray::arr1(&[0.0, 1.0, 10.0]));
        assert_eq!((x * y).eval(), z.eval());

        let x = expr(2.0);
        let y = expr(ndarray::arr1(&[0.0, 1.0,  2.0]));
        let z = expr(ndarray::arr1(&[0.0, 2.0, 4.0]));
        assert_eq!((x.clone() * y.clone()).eval(), z.eval());
        assert_eq!((y * x).eval(), z.eval());

        let x = v("x", Rc::new(VariableValue::new(ndarray::arr1(&[0.0, 1.0, 2.0]))));
        let y = v("y", Rc::new(VariableValue::new(ndarray::arr1(&[0.0, 1.0, 5.0]))));
        assert_eq!((x * y).gradient_by_scalar("x", &ndarray::Ix1(2).into_dyn()).eval(), ndarray::arr1(&[0.0, 0.0, 5.0]).into_dyn());
    }
}
