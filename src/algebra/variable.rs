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

#[derive(Clone)]
pub struct Variable {
    pub is_dx: bool,
    pub name: String,
    pub value: Rc<VariableValue>,
}

impl ExprImpl for Variable {
    fn gradient(&self, v: &str) -> Expr {
        if v == self.name {
            // a variable is left in place so the chain rule can be used to complete the gradient
            Expr::new(Variable{
                is_dx: true,
                name: self.name.clone(),
                value: self.value.clone(),
            })
        } else {
            Expr::new(Constant{
                value: ndarray::Array::zeros((*self.value).shape()),
            })
        }
    }

    fn eval(&self) -> ndarray::ArrayD<f32> {
        (*self.value).get()
    }

    fn shape(&self) -> ndarray::IxDyn {
        (*self.value).shape()
    }

    fn is_constant(&self) -> bool {
        false
    }

    fn propagate_constants(&self) -> Expr {
        Expr::new(self.clone())
    }

    fn freeze_dx(&self, v: &str, i: &ndarray::IxDyn) -> Expr {
        if v == self.name && self.is_dx {
            let mut value = ndarray::Array::zeros((*self.value).shape());
            value[i] = 1.0;
            super::expr(value)
        } else {
            Expr::new(self.clone())
        }
    }
}

impl fmt::Display for Variable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_dx {
            write!(f, "f'({})", self.name)
        } else {
            write!(f, "{}", self.name)
        }
    }
}

pub fn v<T: Into<String>>(name: T, init: Rc<VariableValue>) -> Expr {
    Expr::new(Variable{
        is_dx: false,
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
        assert_eq!(format!("{}", x.gradient_by_scalar("x", &ndarray::Ix0().into_dyn())), "1");

        let x = v("x", Rc::new(VariableValue::new(ndarray::Array::zeros(3))));
        assert_eq!(x.gradient_by_scalar("x", &ndarray::Ix1(1).into_dyn()).eval(), ndarray::arr1(&[0.0, 1.0, 0.0]).into_dyn());
    }
}
