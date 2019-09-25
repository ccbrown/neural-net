use std::fmt;

use ndarray::Dimension;

use super::{Expr, ExprImpl};

// Div performs element-wise division. If the numerator and denominator are not the same shape, one
// must be a scalar.
pub struct Div {
    pub num: Expr,
    pub den: Expr,
}

impl ExprImpl for Div {
    fn gradient(&self, v: &str) -> Expr {
        let num = self.num.gradient(v) * self.den.clone() - self.den.gradient(v) * self.num.clone();
        let den = self.den.clone() * self.den.clone();
        num / den
    }

    fn eval(&self) -> ndarray::ArrayD<f32> {
        let num = self.num.eval();
        let den = self.den.eval();
        if num.ndim() == den.ndim() {
            num / den
        } else if num.ndim() == 0 {
            let shape = den.dim();
            ndarray::Array::from_elem(shape, *num.first().unwrap()) / den
        } else {
            let shape = num.dim();
            num / den.broadcast(shape).unwrap()
        }
    }

    fn shape(&self) -> ndarray::IxDyn {
        let num = self.num.shape();
        if num.ndim() != 0 {
            num
        } else {
            self.den.shape()
        }
    }

    fn is_constant(&self) -> bool {
        if !self.num.is_constant() {
            return false;
        }
        let num = self.num.eval();
        self.den.is_constant() || num == ndarray::Array::zeros(num.dim())
    }

    fn propagate_constants(&self) -> Expr {
        if self.is_constant() {
            super::expr(self.eval())
        } else {
            Expr::new(Self{
                num: self.num.propagate_constants(),
                den: self.den.propagate_constants(),
            })
        }
    }

    fn freeze_dx(&self, v: &str, i: &ndarray::IxDyn) -> Expr {
        Expr::new(Self{
            num: self.num.freeze_dx(v, i),
            den: self.den.freeze_dx(v, i),
        })
    }
}

impl fmt::Display for Div {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({} / {})", self.num, self.den)
    }
}

impl<T: Into<Expr>> std::ops::Div<T> for Expr {
    type Output = Self;
    fn div(self, rhs: T) -> Self {
        Expr::new(Div{
            num: self,
            den: rhs.into(),
        })
    }
}

impl std::ops::Div<Expr> for f32 {
    type Output = Expr;
    fn div(self, rhs: Expr) -> Expr {
        Expr::new(Div{
            num: super::expr(self),
            den: rhs,
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
        assert_eq!(format!("{}", (4.0 / x).gradient_by_scalar("x", &ndarray::Ix0().into_dyn())), "(-4 / (x * x))");

        let x = expr(ndarray::arr1(&[0.0, 1.0, 6.0]));
        let y = expr(ndarray::arr1(&[1.0, 1.0, 2.0]));
        let z = expr(ndarray::arr1(&[0.0, 1.0, 3.0]));
        assert_eq!((x / y).eval(), z.eval());

        let x = expr(2.0);
        let y = expr(ndarray::arr1(&[1.0, 1.0,  2.0]));
        let z = expr(ndarray::arr1(&[2.0, 2.0, 1.0]));
        assert_eq!((x.clone() / y.clone()).eval(), z.eval());
        let z = expr(ndarray::arr1(&[0.5, 0.5, 1.0]));
        assert_eq!((y / x).eval(), z.eval());

        let x = v("x", Rc::new(VariableValue::new(ndarray::arr1(&[0.0, 1.0, 2.0]))));
        let y = v("y", Rc::new(VariableValue::new(ndarray::arr1(&[1.0, 1.0, 5.0]))));
        assert_eq!((x / y).gradient_by_scalar("x", &ndarray::Ix1(2).into_dyn()).eval(), ndarray::arr1(&[0.0, 0.0, 0.2]).into_dyn());
    }
}
