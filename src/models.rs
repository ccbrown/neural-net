use std::error::Error;
use std::cell::Cell;
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

    pub fn compile(self) -> CompiledSequential {
        let input_values = ndarray::Array::from_shape_fn(self.input_shape, |_| Rc::new(Cell::new(0.0)));
        let mut i = 0;
        let input = input_values.mapv(|v| {
            let ret = algebra::v(format!("i{}", i), v);
            i += 1;
            ret
        });
        let mut output = input.clone();
        let mut trainable_variables = Vec::new();
        for (i, layer) in self.layers.iter().enumerate() {
            let instance = layer.init(format!("l{}", i).as_str());
            trainable_variables.extend_from_slice(instance.trainable_variables());
            output = instance.expression(output.view());
        }
        // TODO: actual loss formula
        let loss = output.fold(algebra::c(0.0), |loss, e| loss + e.clone());
        CompiledSequential{
            input: input_values,
            loss: loss,
            trainable_variables: trainable_variables,
        }
    }
}

pub struct CompiledSequential {
    input: ndarray::ArrayD<Rc<Cell<f32>>>,
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
                let input = dataset.input(j)?;
                for (i, v) in self.input.indexed_iter_mut() {
                    (*v).set(input[i]);
                }
                println!("{}: {}", j, self.loss.eval());
            }
        }
        Ok(())
    }
}
