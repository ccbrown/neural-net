use std::fmt;
use std::cell::RefCell;
use std::rc::Rc;

use super::{Constant, Expr, ExprImpl};

pub struct VariableValue(RefCell<ndarray::ArrayD<f32>>);

impl VariableValue {
    pub fn new<S, D>(a: ndarray::ArrayBase<S, D>) -> VariableValue
        where S: ndarray::Data<Elem=f32>,
              D: ndarray::Dimension,
    {
        VariableValue(RefCell::new(a.to_owned().into_dyn()))
    }

    pub fn set<S, D>(&self, a: ndarray::ArrayBase<S, D>)
        where S: ndarray::Data<Elem=f32>,
              D: ndarray::Dimension,
    {
        self.0.replace(a.to_owned().into_dyn());
    }

    pub fn shape(&self) -> ndarray::IxDyn {
        self.0.borrow().dim()
    }

    pub fn get(&self) -> ndarray::ArrayD<f32> {
        self.0.borrow().clone()
    }
}

pub struct Variable {
    pub name: String,
    pub value: Rc<VariableValue>,
}

impl ExprImpl for Variable {
    fn gradient(&self, v: &str, i: &ndarray::IxDyn) -> Expr {
        let mut g = ndarray::Array::zeros((*self.value).shape());
        if v == self.name {
            g[i] = 1.0;
        }
        Expr::new(Constant{
            value: g,
        })
    }

    fn eval(&self) -> ndarray::ArrayD<f32> {
        (*self.value).get()
    }

    fn shape(&self) -> ndarray::IxDyn {
        (*self.value).shape()
    }
}

impl fmt::Display for Variable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name)
    }
}

pub fn v<T: Into<String>>(name: T, init: Rc<VariableValue>) -> Expr {
    Expr::new(Variable{
        name: name.into(),
        value: init,
    })
}

#[cfg(test)]
mod tests {
    use super::super::*;

    use ndarray::Dimension;

    #[test]
    fn test() {
        let x = v("x", Rc::new(VariableValue::new(ndarray::arr0(0.0))));
        assert_eq!(format!("{}", x.gradient("x", &ndarray::Ix0().into_dyn())), "1");
    }
}
