use std::collections::HashMap;

use super::algebra::Expr;

// Combines one or more algebraic expressions into a graph for efficient evaluation. Each graph
// node corresponds to one expression or sub-expression. The graph offers two important performance
// benefits:
//
//   1. Each node in the graph owns memory for its inputs and output. This minimizes allocations
//      during evaluation.
//
//   2. The graph ensures that each sub-expression is evaluated exactly once, even when used by
//      multiple top level expressions. Note that this only eliminates redundant work for
//      expressions with the same id. This works well for gradients returned by Expr::gradients,
//      but a more robust implementation would combine work for all equivalent expressions such as
//      "a+b" and "b+a".
pub struct Graph {
    nodes: Vec<Node>,
    expr_to_node_ids: HashMap<usize, usize>,
    generation: usize,
    top_level_node_ids: Vec<usize>,
}

pub struct Node {
    expr: Expr,
    inputs: Vec<ndarray::ArrayD<f32>>,
    input_node_ids: Vec<usize>,
    output: ndarray::ArrayD<f32>,
    output_generation: usize,
}

impl Graph {
    pub fn new() -> Graph {
        Graph {
            nodes: Vec::new(),
            expr_to_node_ids: HashMap::new(),
            generation: 0,
            top_level_node_ids: Vec::new(),
        }
    }

    // Adds an expression to the graph and returns its new node id. If the expression is already
    // part of the graph, the existing node id is returned.
    pub fn add<E: Into<Expr>>(&mut self, expr: E) -> usize {
        self.add_impl(expr, true)
    }

    fn add_impl<E: Into<Expr>>(&mut self, expr: E, top: bool) -> usize {
        let expr = expr.into();
        match self.expr_to_node_ids.get(&expr.id()) {
            Some(&id) => id,
            None => {
                let input_node_ids = expr
                    .inputs()
                    .iter()
                    .map(|&input| self.add_impl(input.clone(), false))
                    .collect();
                let id = self.nodes.len();
                self.expr_to_node_ids.insert(expr.id(), id);
                let node = Node {
                    inputs: expr
                        .inputs()
                        .iter()
                        .map(|input| ndarray::Array::zeros(input.shape()))
                        .collect(),
                    input_node_ids: input_node_ids,
                    output: ndarray::Array::zeros(expr.shape()),
                    expr: expr,
                    output_generation: 0,
                };
                self.nodes.push(node);
                if top {
                    self.top_level_node_ids.push(id);
                }
                id
            }
        }
    }

    pub fn eval(&mut self) {
        self.eval_nodes(self.top_level_node_ids.clone());
    }

    pub fn eval_nodes(&mut self, mut ids: Vec<usize>) {
        self.generation += 1;
        let mut tmp = Vec::new();
        while let Some(&id) = ids.last() {
            let mut is_ready = true;

            {
                let node = &self.nodes[id];
                if node.output_generation == self.generation {
                    ids.pop();
                    continue;
                }

                for &id in node.input_node_ids.iter() {
                    let input = &self.nodes[id];
                    if input.output_generation != self.generation {
                        is_ready = false;
                        ids.push(id);
                    }
                }
            }

            if is_ready {
                std::mem::swap(&mut self.nodes[id].inputs, &mut tmp);
                for (i, &id) in self.nodes[id].input_node_ids.iter().enumerate() {
                    tmp[i].assign(&self.nodes[id].output);
                }
                let node = &mut self.nodes[id];
                node.output = node.expr.eval_inputs(&tmp);
                node.output_generation = self.generation;
                std::mem::swap(&mut node.inputs, &mut tmp);
                ids.pop();
            }
        }
    }

    pub fn node_output(&self, id: usize) -> &ndarray::ArrayD<f32> {
        &self.nodes[id].output
    }
}
