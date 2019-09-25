use std::fmt;
use std::rc::Rc;

use ndarray::Dimension;

use super::{Expr, ExprImpl, VariableValue};

// Add performs element-wise addition. If the numerator and denominator are not the same shape, one
// must be a scalar.
pub struct Add {
    pub left: Expr,
    pub right: Expr,
}

impl ExprImpl for Add {
    fn gradient(&self, v: &str, fv: &Rc<VariableValue>) -> Expr {
        self.left.gradient(v, fv) + self.right.gradient(v, fv)
    }

    fn eval(&self) -> ndarray::ArrayD<f32> {
        // ndarray will broadcast a scalar into the larger operand, but only if it's on the right
        if self.left.shape().ndim() == 0 {
            self.right.eval() + self.left.eval()
        } else {
            self.left.eval() + self.right.eval()
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
        self.left.is_constant() && self.right.is_constant()
    }

    fn propagate_constants(&self) -> Expr {
        if self.is_constant() {
            super::expr(self.eval())
        } else {
            if self.left.is_constant() {
                let left = self.left.eval();
                if left == ndarray::Array::zeros(left.dim()) {
                    return self.right.propagate_constants();
                }
            }
            if self.right.is_constant() {
                let right = self.right.eval();
                if right == ndarray::Array::zeros(right.dim()) {
                    return self.left.propagate_constants();
                }
            }
            return Expr::new(Self{
                left: self.left.propagate_constants(),
                right: self.right.propagate_constants(),
            });
        }
    }

    fn freeze_variable(&self, name: &str) -> Expr {
        Expr::new(Self{
            left: self.left.freeze_variable(name),
            right: self.right.freeze_variable(name),
        })
    }
}

impl fmt::Display for Add {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({} + {})", self.left, self.right)
    }
}

impl<T: Into<Expr>> std::ops::Add<T> for Expr {
    type Output = Self;
    fn add(self, rhs: T) -> Self {
        Expr::new(Add{
            left: self,
            right: rhs.into(),
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
        let y = v("y", Rc::new(VariableValue::new(ndarray::arr0(0.0))));
        assert_eq!((x.clone() + y).gradient_by_scalar(&x, &ndarray::Ix0().into_dyn()).eval(), ndarray::arr0(1.0).into_dyn());

        let x = v("x", Rc::new(VariableValue::new(ndarray::arr0(0.0))));
        let y = v("y", Rc::new(VariableValue::new(ndarray::arr0(0.0))));
        assert_eq!((x + y.clone()).gradient_by_scalar(&y, &ndarray::Ix0().into_dyn()).eval(), ndarray::arr0(1.0).into_dyn());

        let x = expr(ndarray::arr1(&[0.0, 1.0, 2.0]));
        let y = expr(ndarray::arr1(&[0.0, 1.0, 5.0]));
        let z = expr(ndarray::arr1(&[0.0, 2.0, 7.0]));
        assert_eq!((x + y).eval(), z.eval());

        let x = expr(ndarray::arr0(1.0));
        let y = expr(ndarray::arr1(&[0.0, 1.0, 5.0]));
        let z = expr(ndarray::arr1(&[1.0, 2.0, 6.0]));
        assert_eq!((x + y).eval(), z.eval());

        let x = expr(ndarray::arr1(&[0.0, 1.0, 5.0]));
        let y = expr(ndarray::arr0(1.0));
        let z = expr(ndarray::arr1(&[1.0, 2.0, 6.0]));
        assert_eq!((x + y).eval(), z.eval());

        let x = v("x", Rc::new(VariableValue::new(ndarray::arr1(&[0.0, 1.0, 2.0]))));
        let y = expr(ndarray::arr1(&[0.0, 1.0, 5.0]));
        assert_eq!((x.clone() + y).gradient_by_scalar(&x, &ndarray::Ix1(2).into_dyn()).eval(), ndarray::arr1(&[0.0, 0.0, 1.0]).into_dyn());
    }
}
