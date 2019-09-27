use std::error::Error;
use std::rc::Rc;

use rand::SeedableRng;
use rand::seq::SliceRandom;

use super::{algebra, Dataset, graph, Layer};

// Sequential is used to build a neural network based on layers that are activated in sequence.
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

    // Once the model is final, it needs to be "compiled" before it can do anything. This just does
    // a bit of math up front before returning the object that can be used for training or
    // inference.
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
                    gradient_node_id: 0,
                });
            }
            output = instance.expression(output);
        }
        let target_value = Rc::new(algebra::VariableValue::new(ndarray::Array::zeros(target_shape)));
        let target = algebra::v("t", target_value.clone());
        variable_names.push("t".to_string());
        let loss = loss_function(output.clone(), target);
        let mut graph = graph::Graph::new();
        let gradients = loss.gradients();
        for tv in trainable_variables.iter_mut() {
            tv.gradient_node_id = graph.add(gradients.get(&tv.name).unwrap().clone());
        }
        let output_node_id = graph.add(output);
        CompiledSequential{
            input: input_value,
            target: target_value,
            trainable_variables: trainable_variables,
            graph: graph,
            output_node_id: output_node_id,
        }
    }
}

struct TrainableVariable {
    name: String,
    value: Rc<algebra::VariableValue>,
    gradient_node_id: usize,
}

pub struct CompiledSequential {
    input: Rc<algebra::VariableValue>,
    target: Rc<algebra::VariableValue>,
    trainable_variables: Vec<TrainableVariable>,
    graph: graph::Graph,
    output_node_id: usize,
}

fn max_index<S, D>(a: &ndarray::ArrayBase<S, D>) -> usize
    where S: ndarray::Data<Elem=f32>,
          D: ndarray::Dimension,
{
    let mut selection = 0;
    let mut selection_score = 0.0;
    for (i, &v) in a.view().into_dimensionality::<ndarray::Ix1>().unwrap().indexed_iter() {
        if v > selection_score {
            selection = i;
            selection_score = v;
        }
    }
    selection
}

impl CompiledSequential {
    // Trains the model using stochastic gradient descent.
    pub fn fit<D: Dataset>(&mut self, dataset: &mut D, learning_rate: f32, epochs: usize) -> Result<(), Box<Error>> {
        let mut rng = rand::rngs::StdRng::seed_from_u64(0);
        let gradient_node_ids: Vec<_> = self.trainable_variables.iter().map(|v| v.gradient_node_id).collect();
        for epoch in 0..epochs {
            let mut samples: Vec<usize> = (0..dataset.len()).collect();
            samples.shuffle(&mut rng);
            let mut step = 0;
            for j in samples {
                (*self.input).set(dataset.input(j)?);
                (*self.target).set(dataset.target(j)?);
                self.graph.eval_nodes(gradient_node_ids.clone());
                for tv in self.trainable_variables.iter() {
                    tv.value.set(tv.value.get() - self.graph.node_output(tv.gradient_node_id) * learning_rate);
                }
                if step % 10000 == 0 {
                    info!("epoch {}, step {}; accuracy: {}", epoch, step, self.eval_accuracy(dataset)?);
                }
                step += 1
            }
        }
        Ok(())
    }

    pub fn predict<S, D>(&mut self, input: ndarray::ArrayBase<S, D>) -> &ndarray::ArrayD<f32> 
        where S: ndarray::Data<Elem=f32>,
              D: ndarray::Dimension,
    {
        (*self.input).set(input);
        self.graph.eval_nodes(vec![self.output_node_id]);
        self.graph.node_output(self.output_node_id)
    }

    fn eval_accuracy<D: Dataset>(&mut self, dataset: &mut D) -> Result<f32, Box<Error>> {
        let mut samples: Vec<usize> = (0..dataset.len()).collect();
        samples.shuffle(&mut rand::thread_rng());
        let mut correct = 0;
        for &j in samples.iter().take(std::cmp::max(samples.len(), 500)) {
            if max_index(self.predict(dataset.input(j)?)) == max_index(&dataset.target(j)?) {
                correct += 1
            }
        }
        Ok(correct as f32 / dataset.len() as f32)
    }
}
