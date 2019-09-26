use std::fmt;

use super::{Expr, ExprImpl};

use ndarray::Dimension;

// MatVecMul performs matrix-vector multiplication.
pub struct MatVecMul {
    pub a: Expr,
    pub b: Expr,
}

impl ExprImpl for MatVecMul {
    fn eval(&self) -> ndarray::ArrayD<f32> {
        let a = self.a.eval().into_dimensionality::<ndarray::Ix2>().unwrap();
        let b = self.b.eval().into_dimensionality::<ndarray::Ix1>().unwrap();
        let mut result = ndarray::Array::zeros(a.rows());
        ndarray::linalg::general_mat_vec_mul(1.0, &a, &b, 0.0, &mut result);
        result.into_dyn()
    }

    fn shape(&self) -> ndarray::IxDyn {
        let rows = self.a.shape().as_array_view()[0];
        ndarray::Ix1(rows).into_dyn()
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
            Expr::new(Self{
                a: self.a.propagate_constants(),
                b: self.b.propagate_constants(),
            })
        }
    }

    fn accumulate_gradients(&self, output: Expr, gradients: &mut super::Gradients) {
        let output2 = output.reshape(ndarray::Ix2(output.shape().size(), 1));
        let b2 = self.b.reshape(ndarray::Ix2(1, self.b.shape().size()));
        self.a.accumulate_gradients(super::matmul(output2, b2), gradients);
        self.b.accumulate_gradients(matvecmul(self.a.transpose(), output.clone()), gradients);
    }
}

impl fmt::Display for MatVecMul {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({} · {})", self.a, self.b)
    }
}

pub fn matvecmul<A: Into<Expr>, B: Into<Expr>>(a: A, b: B) -> Expr {
    Expr::new(MatVecMul{
        a: a.into(),
        b: b.into(),
    })
}

#[cfg(test)]
mod tests {
    use super::super::*;

    #[test]
    fn test() {
        let x = v("x", Rc::new(VariableValue::new(ndarray::arr2(&[[0.0, 1.0], [2.0, 3.0]]))));
        let y = v("y", Rc::new(VariableValue::new(ndarray::arr1(&[0.0, 1.0]))));
        println!("{}", matvecmul(x.clone(), y.clone()).gradient("x"));
        assert_eq!(matvecmul(x.clone(), y.clone()).gradient("x").eval(), ndarray::arr2(&[[0.0, 1.0], [0.0, 1.0]]).into_dyn());
        assert_eq!(matvecmul(x.clone(), y.clone()).gradient("y").eval(), ndarray::arr1(&[2.0, 4.0]).into_dyn());

        let x = v("x", Rc::new(VariableValue::new(ndarray::arr2(&[[0.0, 1.0], [2.0, 3.0]]))));
        let y = expr(ndarray::arr1(&[3.0, 5.0]));
        assert_eq!(matvecmul(x.clone(), y).gradient("x").eval(), ndarray::arr2(&[[3.0, 5.0], [3.0, 5.0]]).into_dyn());

        let x = v("x", Rc::new(VariableValue::new(ndarray::arr2(&[[0.0, 1.0], [2.0, 3.0]]))));
        let y = v("y", Rc::new(VariableValue::new(ndarray::arr1(&[3.0, 5.0]))));
        assert_eq!(matvecmul(x.clone(), matvecmul(x.clone(), y.clone())).gradient("x").eval(), ndarray::arr2(&[[11.0, 31.0], [17.0, 41.0]]).into_dyn());
        assert_eq!(matvecmul(x.clone(), matvecmul(x.clone(), y.clone())).gradient("y").eval(), ndarray::arr1(&[8.0, 14.0]).into_dyn());
    }
}
