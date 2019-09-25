use std::fmt;
use std::rc::Rc;

use super::{Expr, ExprImpl, VariableValue};

pub struct Constant {
    pub value: ndarray::ArrayD<f32>,
}

impl ExprImpl for Constant {
    fn gradient(&self, _v: &str, _fv: &Rc<VariableValue>) -> Expr {
        Expr::new(Constant{
            value: ndarray::Array::zeros(self.value.dim()),
        })
    }

    fn eval(&self) -> ndarray::ArrayD<f32> {
        self.value.clone()
    }

    fn shape(&self) -> ndarray::IxDyn {
        self.value.dim()
    }

    fn is_constant(&self) -> bool {
        true
    }

    fn propagate_constants(&self) -> Expr {
        super::expr(self.eval())
    }

    fn freeze_variable(&self, _name: &str) -> Expr {
        super::expr(self.eval())
    }
}

impl fmt::Display for Constant {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.value)
    }
}

impl From<f32> for Expr {
    fn from(f: f32) -> Expr {
        Expr::new(Constant{
            value: ndarray::arr0(f).into_dyn(),
        })
    }
}

impl<S1, D> From<ndarray::ArrayBase<S1, D>> for Expr
    where S1: ndarray::Data<Elem=f32>,
          D: ndarray::Dimension,
{
    fn from(a: ndarray::ArrayBase<S1, D>) -> Expr {
        Expr::new(Constant{
            value: a.into_owned().into_dyn(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::super::*;

    use ndarray::Dimension;

    #[test]
    fn test() {
        let x = v("x", Rc::new(VariableValue::new(ndarray::arr1(&[0.0, 1.0, 2.0]))));
        assert_eq!(format!("{}", expr(99.0).gradient_by_scalar(&x, &ndarray::Ix1(2).into_dyn())), "0");
    }
}
