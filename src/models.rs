use std::error::Error;
use std::rc::Rc;

use rand::SeedableRng;
use rand::seq::SliceRandom;

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
                    gradient: algebra::expr(0.0),
                });
            }
            output = instance.expression(output);
        }
        let target_value = Rc::new(algebra::VariableValue::new(ndarray::Array::zeros(target_shape)));
        let target = algebra::v("t", target_value.clone());
        variable_names.push("t".to_string());
        let loss = loss_function(output.clone(), target);
        for tv in trainable_variables.iter_mut() {
            tv.gradient = loss.gradient(tv.name.as_str());
        }
        CompiledSequential{
            input: input_value,
            output: output,
            target: target_value,
            trainable_variables: trainable_variables,
        }
    }
}

struct TrainableVariable {
    name: String,
    value: Rc<algebra::VariableValue>,
    gradient: algebra::Expr,
}

pub struct CompiledSequential {
    input: Rc<algebra::VariableValue>,
    output: algebra::Expr,
    target: Rc<algebra::VariableValue>,
    trainable_variables: Vec<TrainableVariable>,
}

fn max_index<S, D>(a: ndarray::ArrayBase<S, D>) -> usize
    where S: ndarray::Data<Elem=f32>,
          D: ndarray::Dimension,
{
    let mut selection = 0;
    let mut selection_score = 0.0;
    for (i, &v) in a.into_dimensionality::<ndarray::Ix1>().unwrap().indexed_iter() {
        if v > selection_score {
            selection = i;
            selection_score = v;
        }
    }
    selection
}

impl CompiledSequential {
    pub fn fit<D: Dataset>(&mut self, dataset: &mut D, learning_rate: f32, epochs: usize) -> Result<(), Box<Error>> {
        let mut rng = rand::rngs::StdRng::seed_from_u64(0);
        for epoch in 0..epochs {
            let mut samples: Vec<usize> = (0..dataset.len()).collect();
            samples.shuffle(&mut rng);
            let mut step = 0;
            for j in samples {
                (*self.input).set(dataset.input(j)?);
                (*self.target).set(dataset.target(j)?);
                for tv in self.trainable_variables.iter() {
                    tv.value.set(tv.value.get() - tv.gradient.eval() * learning_rate);
                }
                if step % 1000 == 0 {
                    info!("approx training set accuracy after epoch {}, step {}: {}", epoch, step, self.eval_accuracy(dataset)?);
                }
                step += 1
            }
        }
        Ok(())
    }

    fn eval_accuracy<D: Dataset>(&self, dataset: &mut D) -> Result<f32, Box<Error>> {
        let mut samples: Vec<usize> = (0..dataset.len()).collect();
        samples.shuffle(&mut rand::thread_rng());
        let mut correct = 0;
        for &j in samples.iter().take(std::cmp::max(samples.len(), 500)) {
            (*self.input).set(dataset.input(j)?);
            if max_index(self.output.eval()) == max_index(dataset.target(j)?) {
                correct += 1
            }
        }
        Ok(correct as f32 / dataset.len() as f32)
    }
}
