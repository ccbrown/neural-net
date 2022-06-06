use std::fmt;

use super::{Expr, ExprImpl};

use ndarray::Dimension;

// MatMul performs matrix-matrix multiplication.
pub struct MatMul {
    pub a: Expr,
    pub b: Expr,
}

impl ExprImpl for MatMul {
    fn eval_inputs(&self, inputs: &Vec<ndarray::ArrayD<f32>>) -> ndarray::ArrayD<f32> {
        let (a, b) = (&inputs[0], &inputs[1]);
        let a = a.clone().into_dimensionality::<ndarray::Ix2>().unwrap();
        let b = b.clone().into_dimensionality::<ndarray::Ix2>().unwrap();
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

    fn accumulate_gradients(
        &self,
        output: Expr,
        _gradients: &mut super::Gradients,
    ) -> Vec<Option<Expr>> {
        vec![
            Some(matmul(output.clone(), self.b.transpose())),
            Some(matmul(self.a.transpose(), output.clone())),
        ]
    }

    fn inputs(&self) -> Vec<&Expr> {
        vec![&self.a, &self.b]
    }
}

impl fmt::Display for MatMul {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({} · {})", self.a, self.b)
    }
}

pub fn matmul<A: Into<Expr>, B: Into<Expr>>(a: A, b: B) -> Expr {
    Expr::new(MatMul {
        a: a.into(),
        b: b.into(),
    })
}
