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
        let mut variable_names = vec![];
        let input_value = Rc::new(algebra::VariableValue::new(ndarray::Array::zeros(self.input_shape)));
        let input = algebra::v("i", input_value.clone());
        variable_names.push("i".to_string());
        let mut output = input.clone();
        let mut trainable_variables = Vec::new();
        for (i, layer) in self.layers.iter().enumerate() {
            let instance = layer.init(format!("l{}", i).as_str());
            for v in instance.variables() {
                variable_names.push(v.name.clone());
                trainable_variables.push(TrainableVariable{
                    name: v.name.clone(),
                    value: v.value.clone(),
                    gradient: algebra::expr(0.0), // updated below
                    gradient_fv: Rc::new(algebra::VariableValue::new(ndarray::Array::zeros(v.value.shape()))),
                });
            }
            output = instance.expression(output);
        }
        let target_value = Rc::new(algebra::VariableValue::new(ndarray::Array::zeros(target_shape)));
        let target = algebra::v("t", target_value.clone());
        variable_names.push("t".to_string());
        let loss = loss_function(output, target);
        for tv in trainable_variables.iter_mut() {
            tv.gradient = loss.gradient(tv.name.as_str(), &tv.gradient_fv);
        }
        CompiledSequential{
            input: input_value,
            target: target_value,
            loss: loss,
            trainable_variables: trainable_variables,
            variable_names: variable_names,
        }
    }
}

struct TrainableVariable {
    name: String,
    value: Rc<algebra::VariableValue>,
    gradient: algebra::Expr,
    gradient_fv: Rc<algebra::VariableValue>,
}

pub struct CompiledSequential {
    input: Rc<algebra::VariableValue>,
    target: Rc<algebra::VariableValue>,
    loss: algebra::Expr,
    trainable_variables: Vec<TrainableVariable>,
    variable_names: Vec<String>,
}

impl CompiledSequential {
    pub fn fit<D: Dataset>(&mut self, dataset: &mut D, learning_rate: f32, epochs: usize) -> Result<(), Box<Error>> {
        for i in 0..epochs {
            info!("epoch {}", i);
            // TODO: shuffle
            for j in 0..dataset.len() {
                (*self.input).set(dataset.input(j)?);
                (*self.target).set(dataset.target(j)?);
                let mut gradients = Vec::new();
                for tv in self.trainable_variables.iter() {
                    let mut gradient = tv.gradient.clone();
                    for name in self.variable_names.iter() {
                        gradient = gradient.freeze_variable(name.as_str());
                    }
                    gradient = gradient.simplified();
                    let mut dx = ndarray::Array::zeros((*tv.value).shape());
                    gradients.push(ndarray::Array::from_shape_fn(dx.dim(), |i| {
                        dx[&i] = 1.0;
                        (*tv.gradient_fv).set(dx.clone());
                        let g = gradient.eval();
                        dx[i] = 0.0;
                        *g.first().unwrap()
                    }));
                }
                for (i, tv) in self.trainable_variables.iter_mut().enumerate() {
                    tv.value.set(tv.value.get() - &gradients[i] * learning_rate);
                }
                info!("loss after step {}: {}", j, self.loss.eval());
            }
        }
        Ok(())
    }
}
