use std::fmt;

use super::{Expr, ExprImpl};

pub struct Constant {
    pub value: ndarray::ArrayD<f32>,
}

impl ExprImpl for Constant {
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

    fn accumulate_gradients(&self, _output: Expr, _gradients: &mut super::Gradients) {
        // do nothing
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
