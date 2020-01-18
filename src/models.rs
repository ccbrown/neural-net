use std::error::Error;
use std::rc::Rc;

use rand::seq::SliceRandom;
use rand::SeedableRng;

use super::{algebra, graph, Dataset, Layer};

// Sequential is used to build a neural network based on layers that are activated in sequence.
pub struct Sequential {
    input_shape: ndarray::IxDyn,
    layers: Vec<Box<dyn Layer>>,
}

impl Sequential {
    pub fn new<D: ndarray::Dimension>(input_shape: D) -> Sequential {
        Sequential {
            input_shape: input_shape.clone().into_dyn(),
            layers: Vec::new(),
        }
    }

    pub fn add_layer<L: Layer + 'static>(&mut self, layer: L) -> Result<(), Box<dyn Error>> {
        self.layers.push(Box::new(layer));
        Ok(())
    }

    pub fn compile_for_inference(mut self) -> CompiledInferenceSequential {
        let input_value = Rc::new(algebra::VariableValue::new(ndarray::Array::zeros(
            self.input_shape,
        )));
        let input = algebra::v("i", input_value.clone());
        let mut output = input.clone();
        for (i, layer) in self.layers.drain(..).enumerate() {
            let instance = layer.init(format!("l{}", i).as_str(), &output.shape());
            output = instance.expression(output);
        }
        let mut graph = graph::Graph::new();
        let output_node_id = graph.add(output);
        CompiledInferenceSequential {
            input: input_value,
            graph: graph,
            output_node_id: output_node_id,
        }
    }

    // Once the model is final, it needs to be "compiled" before it can do much. This just does a
    // bit of math up front before returning the object that can be used for training or inference.
    pub fn compile_for_training<D, L>(
        mut self,
        target_shape: D,
        loss_function: L,
    ) -> CompiledTrainingSequential
    where
        D: ndarray::Dimension,
        L: Fn(algebra::Expr, algebra::Expr) -> algebra::Expr + 'static,
    {
        let input_value = Rc::new(algebra::VariableValue::new(ndarray::Array::zeros(
            self.input_shape,
        )));
        let input = algebra::v("i", input_value.clone());
        let mut output = input.clone();
        let mut trainable_variables = Vec::new();
        for (i, layer) in self.layers.drain(..).enumerate() {
            let instance = layer.init(format!("l{}", i).as_str(), &output.shape());
            for v in instance.variables() {
                trainable_variables.push(TrainableVariable {
                    name: v.name.clone(),
                    value: v.value.clone(),
                    gradient_node_id: 0,
                });
            }
            output = instance.expression(output);
        }
        let target_value = Rc::new(algebra::VariableValue::new(ndarray::Array::zeros(
            target_shape,
        )));
        let target = algebra::v("t", target_value.clone());
        let loss = loss_function(output.clone(), target);
        let mut graph = graph::Graph::new();
        let gradients = loss.gradients();
        for tv in trainable_variables.iter_mut() {
            tv.gradient_node_id = graph.add(gradients.get(&tv.name).unwrap().clone());
        }
        let output_node_id = graph.add(output);
        CompiledTrainingSequential {
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

pub struct CompiledInferenceSequential {
    input: Rc<algebra::VariableValue>,
    graph: graph::Graph,
    output_node_id: usize,
}

impl CompiledInferenceSequential {
    pub fn predict<S, D>(&mut self, input: ndarray::ArrayBase<S, D>) -> &ndarray::ArrayD<f32>
    where
        S: ndarray::Data<Elem = f32>,
        D: ndarray::Dimension,
    {
        (*self.input).set(input);
        self.graph.eval_nodes(vec![self.output_node_id]);
        self.graph.node_output(self.output_node_id)
    }
}

pub struct CompiledTrainingSequential {
    input: Rc<algebra::VariableValue>,
    target: Rc<algebra::VariableValue>,
    trainable_variables: Vec<TrainableVariable>,
    graph: graph::Graph,
    output_node_id: usize,
}

fn max_index<S, D>(a: &ndarray::ArrayBase<S, D>) -> usize
where
    S: ndarray::Data<Elem = f32>,
    D: ndarray::Dimension,
{
    let mut selection = 0;
    let mut selection_score = 0.0;
    for (i, &v) in a
        .view()
        .into_dimensionality::<ndarray::Ix1>()
        .unwrap()
        .indexed_iter()
    {
        if v > selection_score {
            selection = i;
            selection_score = v;
        }
    }
    selection
}

impl CompiledTrainingSequential {
    // Trains the model using stochastic gradient descent.
    pub fn fit<D: Dataset>(
        &mut self,
        dataset: &mut D,
        learning_rate: f32,
        epochs: usize,
    ) -> Result<(), Box<dyn Error>> {
        let mut rng = rand::rngs::StdRng::seed_from_u64(0);
        let gradient_node_ids: Vec<_> = self
            .trainable_variables
            .iter()
            .map(|v| v.gradient_node_id)
            .collect();
        for epoch in 0..epochs {
            let mut samples: Vec<usize> = (0..dataset.len()).collect();
            samples.shuffle(&mut rng);
            let mut step = 0;
            for j in samples {
                (*self.input).set(dataset.input(j)?);
                (*self.target).set(dataset.target(j)?);
                self.graph.eval_nodes(gradient_node_ids.clone());
                for tv in self.trainable_variables.iter() {
                    tv.value.set(
                        tv.value.get()
                            - self.graph.node_output(tv.gradient_node_id) * learning_rate,
                    );
                }
                if step % 10000 == 0 {
                    info!(
                        "epoch {}, step {}; accuracy: {}",
                        epoch,
                        step,
                        self.eval_accuracy(dataset)?
                    );
                }
                step += 1
            }
        }
        Ok(())
    }

    pub fn predict<S, D>(&mut self, input: ndarray::ArrayBase<S, D>) -> &ndarray::ArrayD<f32>
    where
        S: ndarray::Data<Elem = f32>,
        D: ndarray::Dimension,
    {
        (*self.input).set(input);
        self.graph.eval_nodes(vec![self.output_node_id]);
        self.graph.node_output(self.output_node_id)
    }

    fn eval_accuracy<D: Dataset>(&mut self, dataset: &mut D) -> Result<f32, Box<dyn Error>> {
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
