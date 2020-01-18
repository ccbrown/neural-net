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
    fn eval_inputs(&self, inputs: &Vec<ndarray::ArrayD<f32>>) -> ndarray::ArrayD<f32> {
        let (num, den) = (&inputs[0], &inputs[1]);
        if num.ndim() == den.ndim() {
            num / den
        } else if num.ndim() == 0 {
            let shape = den.dim();
            num.broadcast(shape).unwrap().into_owned() / den
        } else {
            let shape = num.dim();
            num.clone() / den.broadcast(shape).unwrap()
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
            self.num.propagate_constants() / self.den.propagate_constants()
        }
    }

    fn accumulate_gradients(
        &self,
        output: Expr,
        _gradients: &mut super::Gradients,
    ) -> Vec<Option<Expr>> {
        vec![
            Some(output.clone() / self.den.clone()),
            Some(output.clone() * (-1.0 * self.num.clone() / self.den.square())),
        ]
    }

    fn inputs(&self) -> Vec<&Expr> {
        vec![&self.num, &self.den]
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
        Expr::new(Div {
            num: self,
            den: rhs.into(),
        })
    }
}

impl std::ops::Div<Expr> for f32 {
    type Output = Expr;
    fn div(self, rhs: Expr) -> Expr {
        Expr::new(Div {
            num: super::expr(self),
            den: rhs,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::super::*;

    #[test]
    fn test() {
        let x = v("x", Rc::new(VariableValue::new(ndarray::arr0(0.0))));
        assert_eq!(
            format!("{}", (4.0 / x.clone()).gradient("x")),
            "(-4 / square(x))"
        );

        let x = expr(ndarray::arr1(&[0.0, 1.0, 6.0]));
        let y = expr(ndarray::arr1(&[1.0, 1.0, 2.0]));
        let z = expr(ndarray::arr1(&[0.0, 1.0, 3.0]));
        assert_eq!((x / y).eval(), z.eval());

        let x = expr(2.0);
        let y = expr(ndarray::arr1(&[1.0, 1.0, 2.0]));
        let z = expr(ndarray::arr1(&[2.0, 2.0, 1.0]));
        assert_eq!((x.clone() / y.clone()).eval(), z.eval());
        let z = expr(ndarray::arr1(&[0.5, 0.5, 1.0]));
        assert_eq!((y / x).eval(), z.eval());

        let x = v(
            "x",
            Rc::new(VariableValue::new(ndarray::arr1(&[0.0, 1.0, 2.0]))),
        );
        let y = v(
            "y",
            Rc::new(VariableValue::new(ndarray::arr1(&[1.0, 1.0, 5.0]))),
        );
        assert_eq!(
            (x.clone() / y.clone()).gradient("x").eval(),
            ndarray::arr1(&[1.0, 1.0, 0.2]).into_dyn()
        );
        assert_eq!(
            (x.clone() / y.clone()).gradient("y").eval(),
            ndarray::arr1(&[0.0, -1.0, -0.08]).into_dyn()
        );

        let x = v(
            "x",
            Rc::new(VariableValue::new(ndarray::arr1(&[1.0, 1.0, 1.0]))),
        );
        assert_eq!(
            (x.clone() / x.sum()).gradient("x").eval(),
            ndarray::arr1(&[0.0, 0.0, 0.0]).into_dyn()
        );
    }
}
