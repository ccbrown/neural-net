use std::fmt;

use super::{Expr, ExprImpl};

pub struct Sub {
    pub left: Expr,
    pub right: Expr,
}

impl ExprImpl for Sub {
    fn gradient(&self, v: &str) -> Expr {
        self.left.gradient(v) - self.right.gradient(v)
    }

    fn eval(&self) -> ndarray::ArrayD<f32> {
        self.left.eval() - self.right.eval()
    }

    fn shape(&self) -> ndarray::IxDyn {
        self.left.shape()
    }

    fn is_constant(&self) -> bool {
        self.left.is_constant() && self.right.is_constant()
    }

    fn propagate_constants(&self) -> Expr {
        if self.is_constant() {
            super::expr(self.eval())
        } else {
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

    fn freeze_dx(&self, v: &str, i: &ndarray::IxDyn) -> Expr {
        Expr::new(Self{
            left: self.left.freeze_dx(v, i),
            right: self.right.freeze_dx(v, i),
        })
    }
}

impl fmt::Display for Sub {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({} - {})", self.left, self.right)
    }
}

impl<T: Into<Expr>> std::ops::Sub<T> for Expr {
    type Output = Self;
    fn sub(self, rhs: T) -> Self {
        Expr::new(Sub{
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
        assert_eq!((x - y).gradient_by_scalar("x", &ndarray::Ix0().into_dyn()).eval(), ndarray::arr0(1.0).into_dyn());

        let x = v("x", Rc::new(VariableValue::new(ndarray::arr0(0.0))));
        let y = v("y", Rc::new(VariableValue::new(ndarray::arr0(0.0))));
        assert_eq!((x - y).gradient_by_scalar("y", &ndarray::Ix0().into_dyn()).eval(), ndarray::arr0(-1.0).into_dyn());

        let x = expr(ndarray::arr1(&[0.0, 1.0,  2.0]));
        let y = expr(ndarray::arr1(&[0.0, 1.0,  5.0]));
        let z = expr(ndarray::arr1(&[0.0, 0.0, -3.0]));
        assert_eq!((x - y).eval(), z.eval());

        let x = v("x", Rc::new(VariableValue::new(ndarray::arr1(&[0.0, 1.0, 2.0]))));
        let y = v("y", Rc::new(VariableValue::new(ndarray::arr1(&[0.0, 1.0, 5.0]))));
        assert_eq!((x - y).gradient_by_scalar("x", &ndarray::Ix1(2).into_dyn()).eval(), ndarray::arr1(&[0.0, 0.0, 1.0]).into_dyn());

        let x = v("x", Rc::new(VariableValue::new(ndarray::arr1(&[0.0, 1.0, 2.0]))));
        let y = v("y", Rc::new(VariableValue::new(ndarray::arr1(&[0.0, 1.0, 5.0]))));
        assert_eq!((x - y).gradient_by_scalar("y", &ndarray::Ix1(2).into_dyn()).eval(), ndarray::arr1(&[0.0, 0.0, -1.0]).into_dyn());
    }
}
