use std::error::Error;
use std::rc::Rc;

use super::{algebra, Dataset, Layer};

pub struct Sequential {
    input_shape: ndarray::IxDyn,
    output_shape: ndarray::IxDyn,
    layers: Vec<Box<Layer>>,
}

impl Sequential {
    pub fn new<D: ndarray::Dimension>(input_shape: D) -> Sequential {
        Sequential{
            input_shape: input_shape.clone().into_dyn(),
            output_shape: input_shape.clone().into_dyn(),
            layers: Vec::new(),
        }
    }

    pub fn output_shape(&self) -> ndarray::IxDyn {
        self.output_shape.clone()
    }

    pub fn add_layer<L: Layer + 'static>(&mut self, layer: L) -> Result<(), Box<Error>> {
        if self.output_shape != layer.input_shape() {
            bail!("invalid input shape for layer. got {:?}, expected {:?}", layer.input_shape(), self.output_shape)
        }

        self.output_shape = layer.output_shape();
        self.layers.push(Box::new(layer));
        Ok(())
    }

    pub fn compile<D, L>(self, target_shape: D, loss_function: L) -> CompiledSequential
        where D: ndarray::Dimension,
              L: Fn(algebra::Expr, algebra::Expr) -> algebra::Expr + 'static,
    {
        let input_value = Rc::new(algebra::VariableValue::new(ndarray::Array::zeros(self.input_shape)));
        let input = algebra::v("i", input_value.clone());
        let mut output = input.clone();
        let mut trainable_variables = Vec::new();
        for (i, layer) in self.layers.iter().enumerate() {
            let instance = layer.init(format!("l{}", i).as_str());
            trainable_variables.extend_from_slice(instance.trainable_variables());
            output = instance.expression(output);
        }
        let target_value = Rc::new(algebra::VariableValue::new(ndarray::Array::zeros(target_shape)));
        let target = algebra::v("t", target_value.clone());
        CompiledSequential{
            input: input_value,
            target: target_value,
            loss: loss_function(output, target),
            trainable_variables: trainable_variables,
        }
    }
}

pub struct CompiledSequential {
    input: Rc<algebra::VariableValue>,
    target: Rc<algebra::VariableValue>,
    loss: algebra::Expr,
    trainable_variables: Vec<super::TrainableVariable>,
}

impl CompiledSequential {
    pub fn fit<D: Dataset>(&mut self, dataset: &mut D, epochs: usize) -> Result<(), Box<Error>> {
        println!("trainable variables: {}", self.trainable_variables.len());
        for i in 0..epochs {
            println!("epoch {}", i);
            // TODO: shuffle
            for j in 0..dataset.len() {
                (*self.input).set(dataset.input(j)?);
                (*self.target).set(dataset.target(j)?);
                println!("{}: {}", j, self.loss.eval());
            }
        }
        Ok(())
    }
}
