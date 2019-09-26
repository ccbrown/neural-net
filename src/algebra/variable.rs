use std::fmt;
use std::cell::RefCell;
use std::ops::DerefMut;
use std::rc::Rc;

use super::{Expr, ExprImpl};

pub struct VariableValue(RefCell<ndarray::ArrayD<f32>>);

impl VariableValue {
    pub fn new<S, D>(a: ndarray::ArrayBase<S, D>) -> VariableValue
        where S: ndarray::Data<Elem=f32>,
              D: ndarray::Dimension,
    {
        VariableValue(RefCell::new(a.into_owned().into_dyn()))
    }

    pub fn set<S, D>(&self, a: ndarray::ArrayBase<S, D>)
        where S: ndarray::Data<Elem=f32>,
              D: ndarray::Dimension,
    {
        self.0.replace(a.into_owned().into_dyn());
    }

    pub fn mutate<F>(&self, f: F)
        where F: FnOnce(&mut ndarray::ArrayD<f32>)
    {
        let mut r = self.0.borrow_mut();
        f(r.deref_mut())
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

    fn accumulate_gradients(&self, output: Expr, gradients: &mut super::Gradients) {
        gradients.expressions.insert(self.name.clone(), match gradients.expressions.get(&self.name) {
            Some(grad) => grad.clone() + output,
            None => output,
        });
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

    #[test]
    fn test() {
        let x = v("x", Rc::new(VariableValue::new(ndarray::arr0(0.0))));
        assert_eq!(x.gradient("x").eval(), ndarray::arr0(1.0).into_dyn());

        let x = v("x", Rc::new(VariableValue::new(ndarray::Array::zeros(3))));
        assert_eq!(x.gradient("x").eval(), ndarray::arr1(&[1.0, 1.0, 1.0]).into_dyn());
    }
}
