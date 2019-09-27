use std::fmt;

use super::{Expr, ExprImpl};

use ndarray::Dimension;

// MatMul performs matrix-matrix multiplication.
pub struct MatMul {
    pub a: Expr,
    pub b: Expr,
}

impl ExprImpl for MatMul {
    fn eval(&self) -> ndarray::ArrayD<f32> {
        let a = self.a.eval().into_dimensionality::<ndarray::Ix2>().unwrap();
        let b = self.b.eval().into_dimensionality::<ndarray::Ix2>().unwrap();
        let mut result = ndarray::Array::zeros((a.rows(), b.cols()));
        ndarray::linalg::general_mat_mul(1.0, &a, &b, 0.0, &mut result);
        result.into_dyn()
    }

    fn shape(&self) -> ndarray::IxDyn {
        let rows = self.a.shape().as_array_view()[0];
        let cols = self.b.shape().as_array_view()[1];
        ndarray::Ix2(rows, cols).into_dyn()
    }

    fn is_constant(&self) -> bool {
        let a = self.a.is_constant();
        let b = self.b.is_constant();
        if a && b {
            true
        } else if a {
            let a = self.a.eval();
            a == ndarray::Array::zeros(a.dim())
        } else if b {
            let b = self.b.eval();
            b == ndarray::Array::zeros(b.dim())
        } else {
            false
        }
    }

    fn propagate_constants(&self) -> Expr {
        if self.is_constant() {
            super::expr(self.eval())
        } else {
            matmul(self.a.propagate_constants(), self.b.propagate_constants())
        }
    }

    fn accumulate_gradients(&self, output: Expr, gradients: &mut super::Gradients) {
        self.a.accumulate_gradients(matmul(output.clone(), self.b.transpose()), gradients);
        self.b.accumulate_gradients(matmul(self.a.transpose(), output.clone()), gradients);
    }
}

impl fmt::Display for MatMul {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({} · {})", self.a, self.b)
    }
}

pub fn matmul<A: Into<Expr>, B: Into<Expr>>(a: A, b: B) -> Expr {
    Expr::new(MatMul{
        a: a.into(),
        b: b.into(),
    })
}
