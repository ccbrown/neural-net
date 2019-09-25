use std::fmt;

use super::{Expr, ExprImpl};

use ndarray::Dimension;

// MatVecMul performs matrix-vector multiplication.
pub struct MatVecMul {
    pub left: Expr,
    pub right: Expr,
}

impl ExprImpl for MatVecMul {
    fn gradient(&self, v: &str) -> Expr {
        matvecmul(self.left.clone(), self.right.gradient(v)) + matvecmul(self.left.gradient(v), self.right.clone())
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
            Expr::new(Self{
                left: self.left.propagate_constants(),
                right: self.right.propagate_constants(),
            })
        }
    }

    fn freeze_dx(&self, v: &str, i: &ndarray::IxDyn) -> Expr {
        Expr::new(Self{
            left: self.left.freeze_dx(v, i),
            right: self.right.freeze_dx(v, i),
        })
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

#[cfg(test)]
mod tests {
    use super::super::*;

    use ndarray::Dimension;

    #[test]
    fn test() {
        let x = expr(ndarray::arr2(&[[0.0, 1.0], [2.0, 3.0]]));
        let y = v("y", Rc::new(VariableValue::new(ndarray::arr1(&[0.0, 1.0]))));
        assert_eq!(matvecmul(x, y).gradient_by_scalar("y", &ndarray::Ix1(1).into_dyn()).eval(), ndarray::arr1(&[1.0, 3.0]).into_dyn());

        let x = v("x", Rc::new(VariableValue::new(ndarray::arr2(&[[0.0, 1.0], [2.0, 3.0]]))));
        let y = expr(ndarray::arr1(&[3.0, 5.0]));
        assert_eq!(matvecmul(x, y).gradient_by_scalar("x", &ndarray::Ix2(1, 0).into_dyn()).eval(), ndarray::arr1(&[0.0, 3.0]).into_dyn());

        // [00 01] · [0] = [00 * 0 + 01 * 1]
        // [10 11]   [1]   [10 * 0 + 11 * 1]
        // [00 01] · [00 * 0 + 01 * 1] = [00 * 00 * 0 + 00 * 01 * 1 + 01 * 10 * 0 + 01 * 11 * 1]
        // [10 11]   [10 * 0 + 11 * 1]   [10 * 00 * 0 + 10 * 01 * 1 + 11 * 10 * 0 + 11 * 11 * 1]
        // d/d10 = [01 * 0         ]
        //         [01 * 1 + 11 * 0]
        let x = v("x", Rc::new(VariableValue::new(ndarray::arr2(&[[0.0, 1.0], [2.0, 3.0]]))));
        let y = expr(ndarray::arr1(&[3.0, 5.0]));
        assert_eq!(matvecmul(x.clone(), matvecmul(x, y)).gradient_by_scalar("x", &ndarray::Ix2(1, 0).into_dyn()).eval(), ndarray::arr1(&[3.0, 14.0]).into_dyn());
    }
}
