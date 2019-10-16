use std::fmt;

use super::{Expr, ExprImpl};

// BiasAdd performs adds a 1-d bias to a value whose last dimension is the same size as the bias
// vector.
pub struct BiasAdd {
    pub value: Expr,
    pub bias: Expr,
}

impl ExprImpl for BiasAdd {
    fn eval_inputs(&self, inputs: &Vec<ndarray::ArrayD<f32>>) -> ndarray::ArrayD<f32> {
        let (value, bias) = (&inputs[0], &inputs[1]);
        value + bias
    }

    fn shape(&self) -> ndarray::IxDyn {
        self.value.shape()
    }

    fn is_constant(&self) -> bool {
        self.value.is_constant() && self.bias.is_constant()
    }

    fn propagate_constants(&self) -> Expr {
        if self.is_constant() {
            super::expr(self.eval())
        } else {
            if self.bias.is_constant() {
                let bias = self.bias.eval();
                if bias == ndarray::Array::zeros(bias.dim()) && self.shape() == self.value.shape() {
                    return self.value.propagate_constants();
                }
            }
            bias_add(self.value.propagate_constants(), self.bias.propagate_constants())
        }
    }

    fn accumulate_gradients(&self, output: Expr, gradients: &mut super::Gradients) {
        self.value.accumulate_gradients(output.clone(), gradients);
        self.bias.accumulate_gradients(output.clone(), gradients);
    }

    fn inputs(&self) -> Vec<&Expr> {
        vec![&self.value, &self.bias]
    }
}

impl fmt::Display for BiasAdd {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "bias_add({}, {})", self.value, self.bias)
    }
}

pub fn bias_add<V: Into<Expr>, B: Into<Expr>>(value: V, bias: B) -> Expr {
    Expr::new(BiasAdd{
        value: value.into(),
        bias: bias.into(),
    })
}

#[cfg(test)]
mod tests {
    use super::super::*;

    #[test]
    fn test() {
        let value = expr(ndarray::arr3(&[[[0.0, 1.0], [1.0, 2.0]], [[2.0, 3.0], [3.0, 4.0]]]));
        let bias = ndarray::arr1(&[1.0, 2.0]);
        assert_eq!(bias_add(value, bias).eval(), ndarray::arr3(&[[[1.0, 3.0], [2.0, 4.0]], [[3.0, 5.0], [4.0, 6.0]]]).into_dyn())
    }
}
