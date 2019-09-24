use std::fmt;

use super::{Expr, ExprImpl};

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
        write!(f, "{}", match self {
            Op::Less => "<",
            Op::LessOrEqual => "<=",
            Op::Greater => ">",
            Op::GreaterOrEqual => ">=",
            Op::Equal => "=",
            Op::NotEqual => "!=",
        })
    }
}

// Cmp performs an element-wise comparison and outputs 1 or 0 based on the result. left and right
// must be the same shape.
pub struct Cmp {
    pub left: Expr,
    pub right: Expr,
    pub op: Op,
}

impl ExprImpl for Cmp {
    fn gradient(&self, _v: &str, _i: &ndarray::IxDyn) -> Expr {
        panic!("gradients are not supported for comparisons")
    }

    fn eval(&self) -> ndarray::ArrayD<f32> {
        let left = self.left.eval();
        let right = self.right.eval();
        let mut result = ndarray::Array::zeros(left.dim());
        ndarray::Zip::from(&mut result)
            .and(&left)
            .and(&right)
            .apply(|out, &a, &b| *out = if self.op.cmp(a, b) { 1.0 } else { 0.0 });
        result
    }

    fn shape(&self) -> ndarray::IxDyn {
        self.left.shape()
    }
}

impl fmt::Display for Cmp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({} {} {})", self.left, self.op, self.right)
    }
}

pub fn cmp(left: Expr, op: Op, right: Expr) -> Expr {
    Expr::new(Cmp{
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
