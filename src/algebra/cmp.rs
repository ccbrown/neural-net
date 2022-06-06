use std::fmt;

use ndarray::Dimension;

use super::{Expr, ExprImpl};

#[derive(Clone)]
pub enum Op {
    Less,
    LessOrEqual,
    Greater,
    GreaterOrEqual,
    Equal,
    NotEqual,
}

impl Op {
    pub fn cmp(&self, left: f32, right: f32) -> bool {
        match self {
            Op::Less => left < right,
            Op::LessOrEqual => left <= right,
            Op::Greater => left > right,
            Op::GreaterOrEqual => left >= right,
            Op::Equal => left == right,
            Op::NotEqual => left != right,
        }
    }
}

impl fmt::Display for Op {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}",
            match self {
                Op::Less => "<",
                Op::LessOrEqual => "<=",
                Op::Greater => ">",
                Op::GreaterOrEqual => ">=",
                Op::Equal => "=",
                Op::NotEqual => "!=",
            }
        )
    }
}

// Cmp performs an element-wise comparison and outputs 1 or 0 based on the result. If left and right
// are not the same shape, one must be a scalar.
pub struct Cmp {
    pub left: Expr,
    pub right: Expr,
    pub op: Op,
}

impl ExprImpl for Cmp {
    fn eval_inputs(&self, inputs: &Vec<ndarray::ArrayD<f32>>) -> ndarray::ArrayD<f32> {
        let (left, right) = (&inputs[0], &inputs[1]);
        if left.ndim() == 0 {
            let left = *left.first().unwrap();
            right.mapv(|v| if self.op.cmp(left, v) { 1.0 } else { 0.0 })
        } else if right.ndim() == 0 {
            let right = *right.first().unwrap();
            left.mapv(|v| if self.op.cmp(v, right) { 1.0 } else { 0.0 })
        } else {
            let mut result = unsafe { ndarray::Array::uninitialized(self.shape()) };
            ndarray::Zip::from(&mut result)
                .and(left)
                .and(right)
                .apply(|o, &l, &r| *o = if self.op.cmp(l, r) { 1.0 } else { 0.0 });
            result
        }
    }

    fn shape(&self) -> ndarray::IxDyn {
        let left = self.left.shape();
        if left.ndim() != 0 {
            left
        } else {
            self.right.shape()
        }
    }

    fn is_constant(&self) -> bool {
        self.left.is_constant() && self.right.is_constant()
    }

    fn propagate_constants(&self) -> Expr {
        if self.is_constant() {
            super::expr(self.eval())
        } else {
            cmp(
                self.left.propagate_constants(),
                self.op.clone(),
                self.right.propagate_constants(),
            )
        }
    }

    fn accumulate_gradients(
        &self,
        _output: Expr,
        _gradients: &mut super::Gradients,
    ) -> Vec<Option<Expr>> {
        panic!("gradients are not supported for comparisons")
    }

    fn inputs(&self) -> Vec<&Expr> {
        vec![&self.left, &self.right]
    }
}

impl fmt::Display for Cmp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({} {} {})", self.left, self.op, self.right)
    }
}

pub fn cmp(left: Expr, op: Op, right: Expr) -> Expr {
    Expr::new(Cmp {
        left: left,
        right: right,
        op: op,
    })
}

#[cfg(test)]
mod tests {
    use super::super::*;

    #[test]
    fn test() {
        let a = expr(ndarray::arr0(0.0));
        let b = expr(ndarray::arr0(1.0));
        let c = ndarray::arr0(1.0);
        assert_eq!(cmp(a, Op::Less, b).eval(), c.into_dyn());

        let a = expr(ndarray::arr1(&[0.0, 2.0]));
        let b = expr(ndarray::arr1(&[1.0, 1.0]));
        let c = ndarray::arr1(&[1.0, 0.0]);
        assert_eq!(cmp(a, Op::Less, b).eval(), c.into_dyn());
    }
}
