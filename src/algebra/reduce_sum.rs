use std::fmt;

use ndarray::Dimension;

use super::{Expr, ExprImpl};

// ReduceSum sums up every element across one or more axes. The output retains the same dimensionality.
pub struct ReduceSum {
    pub expr: Expr,
    pub axes: Vec<usize>,
}

impl ExprImpl for ReduceSum {
    fn eval_inputs(&self, inputs: &Vec<ndarray::ArrayD<f32>>) -> ndarray::ArrayD<f32> {
        let mut result = inputs[0].clone();
        for &axis in self.axes.iter() {
            let axis = ndarray::Axis(axis);
            result = result.map_axis(axis, |a| a.sum()).insert_axis(axis);
        }
        result
    }

    fn shape(&self) -> ndarray::IxDyn {
        let mut ret = self.expr.shape();
        for &axis in self.axes.iter() {
            ret.as_array_view_mut()[axis] = 1;
        }
        ret
    }

    fn is_constant(&self) -> bool {
        self.expr.is_constant()
    }

    fn propagate_constants(&self) -> Expr {
        if self.is_constant() {
            super::expr(self.eval())
        } else {
            reduce_sum(self.expr.propagate_constants(), self.axes.clone())
        }
    }

    fn accumulate_gradients(&self, output: Expr, _gradients: &mut super::Gradients) -> Vec<Option<Expr>> {
        vec![Some(super::broadcast_to(output, self.expr.shape()))]
    }

    fn inputs(&self) -> Vec<&Expr> {
        vec![&self.expr]
    }
}

impl fmt::Display for ReduceSum {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "reduce_sum({}, {:?})", self.expr, self.axes)
    }
}

pub fn reduce_sum<V: Into<Expr>>(expr: V, axes: Vec<usize>) -> Expr {
    Expr::new(ReduceSum{
        expr: expr.into(),
        axes: axes,
    })
}

#[cfg(test)]
mod tests {
    use super::super::*;

    #[test]
    fn test() {
        let x = v("x", Rc::new(VariableValue::new(ndarray::arr2(&[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]))));
        assert_eq!(reduce_sum(x.clone(), vec![1]).eval(), ndarray::arr2(&[[3.0], [3.0]]).into_dyn());

        let x = v("x", Rc::new(VariableValue::new(ndarray::arr2(&[[0.0, 1.0], [2.0, 3.0]]))));
        assert_eq!((reduce_sum(x.clone(), vec![1]) / 2.0).gradient("x").eval(), ndarray::arr2(&[[0.5, 0.5], [0.5, 0.5]]).into_dyn());
    }
}
