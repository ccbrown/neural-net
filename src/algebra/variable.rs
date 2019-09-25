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
    pub name: String,
    pub value: Rc<VariableValue>,
}

impl ExprImpl for Variable {
    fn gradient(&self, v: &str, fv: &Rc<VariableValue>) -> Expr {
        if v == self.name {
            Expr::new(Variable{
                name: "f'".to_string() + self.name.as_str(),
                value: fv.clone(),
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

    fn freeze_variable(&self, name: &str) -> Expr {
        if name == self.name {
            super::expr(self.eval())
        } else {
            Expr::new(self.clone())
        }
    }

    fn variable_name(&self) -> Option<&str> {
        Some(self.name.as_str())
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
        assert_eq!(x.gradient_by_scalar(&x, &ndarray::Ix0().into_dyn()).eval(), ndarray::arr0(1.0).into_dyn());

        let x = v("x", Rc::new(VariableValue::new(ndarray::Array::zeros(3))));
        assert_eq!(x.gradient_by_scalar(&x, &ndarray::Ix1(1).into_dyn()).eval(), ndarray::arr1(&[0.0, 1.0, 0.0]).into_dyn());
    }
}
