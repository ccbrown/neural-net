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

// Cmp performs a comparison and outputs 1 or 0 based on the result.
pub struct Cmp {
    pub left: Expr,
    pub right: Expr,
    pub op: Op,
}

impl ExprImpl for Cmp {
    fn gradient(&self, _v: &str) -> Expr {
        panic!("gradients are not supported for comparisons")
    }

    fn eval(&self) -> f32 {
        if self.op.cmp(self.left.eval(), self.right.eval()) { 1.0 } else { 0.0 }
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
