use std::fmt;

use ndarray::Dimension;

use super::{Expr, ExprImpl};

// BroadcastTo expands one or more axes by repeating elements to match the given shape. An axis can
// be broadcasted if it is either missing or has exactly 1 element.
pub struct BroadcastTo {
    pub expr: Expr,
    pub shape: ndarray::IxDyn,
}

impl ExprImpl for BroadcastTo {
    fn eval_inputs(&self, inputs: &Vec<ndarray::ArrayD<f32>>) -> ndarray::ArrayD<f32> {
        inputs[0].broadcast(self.shape.clone()).unwrap().into_owned()
    }

    fn shape(&self) -> ndarray::IxDyn {
        self.shape.clone()
    }

    fn is_constant(&self) -> bool {
        self.expr.is_constant()
    }

    fn propagate_constants(&self) -> Expr {
        if self.is_constant() {
            super::expr(self.eval())
        } else {
            broadcast_to(self.expr.propagate_constants(), self.shape.clone())
        }
    }

    fn accumulate_gradients(&self, output: Expr, _gradients: &mut super::Gradients) -> Vec<Option<Expr>> {
        let expr_shape = self.expr.shape();
        let missing = self.shape.ndim() - expr_shape.ndim();
        let mut reduction_axes: Vec<_> = (0..missing).collect();
        for i in 0..expr_shape.ndim() {
            if expr_shape[i] == 1 {
                reduction_axes.push(missing + i);
            }
        }
        vec![Some(super::reduce_sum(output, reduction_axes).reshape(expr_shape))]
    }

    fn inputs(&self) -> Vec<&Expr> {
        vec![&self.expr]
    }
}

impl fmt::Display for BroadcastTo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "broadcast_to({}, {:?})", self.expr, self.shape)
    }
}

pub fn broadcast_to<V: Into<Expr>>(expr: V, shape: ndarray::IxDyn) -> Expr {
    Expr::new(BroadcastTo{
        expr: expr.into(),
        shape: shape,
    })
}

#[cfg(test)]
mod tests {
    use super::super::*;

    #[test]
    fn test() {
        let x = v("x", Rc::new(VariableValue::new(ndarray::arr1(&[0.0, 1.0]))));
        assert_eq!(broadcast_to(x.clone(), ndarray::Ix2(3, 2).into_dyn()).eval(), ndarray::arr2(&[[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]).into_dyn());
        assert_eq!(broadcast_to(x.clone(), ndarray::Ix2(3, 2).into_dyn()).gradient("x").eval(), ndarray::arr1(&[3.0, 3.0]).into_dyn());
    }
}
