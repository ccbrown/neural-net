use super::{LayerVariablesBuilder};
use super::super::{algebra, Layer, LayerInstance};

use ndarray::Dimension;

// Dense takes a 1-dimensional input and emits a 1-dimensional output.
pub struct Dense<Activation, KernelInitializer>
    where Activation: Fn(algebra::Expr) -> algebra::Expr + 'static,
          KernelInitializer: Fn(&ndarray::IxDyn) -> ndarray::ArrayD<f32>
{
    pub activation: Activation,
    pub kernel_initializer: KernelInitializer,
    pub output_size: usize,
}

impl<Activation, KernelInitializer> Layer for Dense<Activation, KernelInitializer>
    where Activation: Fn(algebra::Expr) -> algebra::Expr + Clone + 'static,
          KernelInitializer: Fn(&ndarray::IxDyn) -> ndarray::ArrayD<f32>
{
    fn init(&self, namespace: &str, input_shape: &ndarray::IxDyn) -> Box<LayerInstance> {
        let mut lv_builder = LayerVariablesBuilder::new(namespace);
        let activation = self.activation.clone();
        let biases = lv_builder.append("b", ndarray::Array::zeros(self.output_size));
        let weights = lv_builder.append("w", (self.kernel_initializer)(&ndarray::Ix2(self.output_size, input_shape.size()).into_dyn()));
        Box::new(super::Instance{
            expression: move |input| {
                (activation)(algebra::matvecmul(weights.clone(), input) + biases.clone())
            },
            variables: lv_builder.variables,
        })
    }
}

#[cfg(test)]
mod tests {
    use std::rc::Rc;

    use super::*;
    use super::super::super::{activations, initializers, losses};

    #[test]
    fn test() {
        let a = ndarray::Array::range(0.0, 16.0, 1.0).into_dyn();
        assert_eq!(Dense{
            activation: |x| x + 1.0,
            kernel_initializer: initializers::zeros,
            output_size: 4,
        }.init("l", &a.dim()).eval(a.view()), ndarray::Array::ones(4).into_dyn());
    }

    #[test]
    fn test_gradients() {
        let input = algebra::v("i", Rc::new(algebra::VariableValue::new(ndarray::arr1(&[0.0, 1.0, 2.0]))));
        let l = Dense{
            activation: activations::softmax,
            kernel_initializer: initializers::zeros,
            output_size: 3,
        }.init("l", &input.shape()).expression(input);
        let loss = losses::categorical_cross_entropy(l.clone(), algebra::expr(ndarray::arr1(&[0.0, 0.0, 1.0])));

        // import tensorflow as tf
        // sess = tf.compat.v1.Session()
        // input = tf.constant([[0.0, 1.0, 2.0]])
        // l = tf.layers.dense(input, 3, activation=tf.nn.softmax, kernel_initializer=tf.zeros_initializer())
        // loss = -tf.reduce_sum(tf.constant([0.0, 0.0, 1.0]) * tf.log(l))
        // sess.run(tf.compat.v1.global_variables_initializer())

        // sess.run(loss)
        assert_eq!(loss.eval(), ndarray::arr0(1.0986123).into_dyn());

        // sess.run(l)
        assert_eq!(l.eval(), ndarray::arr1(&[0.33333334, 0.33333334, 0.33333334]).into_dyn());

        // sess.run(tf.gradients(l, tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=tf.get_variable_scope().name)))
        assert_eq!(l.gradient("l.b").eval(), ndarray::arr1(&[0.0, 0.0, 0.0]).into_dyn());
        assert_eq!(l.gradient("l.w").eval(), ndarray::arr2(&[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]).into_dyn());

        // sess.run(tf.gradients(loss, tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=tf.get_variable_scope().name)))
        assert_eq!(loss.gradient("l.b").eval(), ndarray::arr1(&[0.33333334, 0.33333334, -0.6666667]).into_dyn());
        assert_eq!(loss.gradient("l.w").eval(), ndarray::arr2(&[[0.0, 0.33333334, 0.6666667], [0.0, 0.33333334, 0.6666667], [0.0, -0.6666667, -1.3333334]]).into_dyn());
    }
}
