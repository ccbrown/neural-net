use std::fmt;
use std::cell::Cell;
use std::rc::Rc;

use super::{Constant, Expr, ExprImpl};

pub struct Variable {
    pub name: String,
    pub value: Rc<Cell<f32>>,
}

impl ExprImpl for Variable {
    fn gradient(&self, v: &str) -> Expr {
        Expr::new(Constant{
            value: if v != self.name { 0.0 } else { 1.0 },
        })
    }

    fn eval(&self) -> f32 {
        (*self.value).get()
    }
}

impl fmt::Display for Variable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name)
    }
}

pub fn v<T: Into<String>>(name: T, init: Rc<Cell<f32>>) -> Expr {
    Expr::new(Variable{
        name: name.into(),
        value: init,
    })
}

#[cfg(test)]
mod tests {
    use super::super::*;

    use std::cell::Cell;

    #[test]
    fn test() {
        let x = v("x", Rc::new(Cell::new(0.0)));
        assert_eq!(format!("{}", x.gradient("x")), "1");
    }
}
