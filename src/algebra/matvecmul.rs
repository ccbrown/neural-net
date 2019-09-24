use std::fmt;

use super::{Expr, ExprImpl};

use ndarray::Dimension;

// MatVecMul performs matrix-vector multiplication.
pub struct MatVecMul {
    pub left: Expr,
    pub right: Expr,
}

impl ExprImpl for MatVecMul {
    fn gradient(&self, v: &str, i: &ndarray::IxDyn) -> Expr {
        // TODO: matrix-by-scalar
        self.left.clone() * self.right.gradient(v, i) + self.right.clone() * self.left.gradient(v, i)
    }

    fn eval(&self) -> ndarray::ArrayD<f32> {
        let left = self.left.eval().into_dimensionality::<ndarray::Ix2>().unwrap();
        let right = self.right.eval().into_dimensionality::<ndarray::Ix1>().unwrap();
        let mut result = ndarray::Array::zeros(left.rows());
        ndarray::linalg::general_mat_vec_mul(1.0, &left, &right, 0.0, &mut result);
        result.into_dyn()
    }

    fn shape(&self) -> ndarray::IxDyn {
        let rows = self.left.shape().as_array_view()[0];
        ndarray::Ix1(rows).into_dyn()
    }
}

impl fmt::Display for MatVecMul {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({} · {})", self.left, self.right)
    }
}

pub fn matvecmul<A: Into<Expr>, B: Into<Expr>>(a: A, b: B) -> Expr {
    Expr::new(MatVecMul{
        left: a.into(),
        right: b.into(),
    })
}
