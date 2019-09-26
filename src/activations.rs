use super::algebra;

pub fn relu(input: algebra::Expr) -> algebra::Expr {
    input.max(algebra::expr(0.0))
}

pub fn softmax(input: algebra::Expr) -> algebra::Expr {
    input.softmax()
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::*;

    #[test]
    fn test_softmax() {
        let x = algebra::v("x", Rc::new(algebra::VariableValue::new(ndarray::arr1(&[0.0, 0.0, 0.0]))));
        let f = softmax(x);

        // import tensorflow as tf
        // sess = tf.compat.v1.Session()
        // x = tf.Variable([0.0, 0.0, 0.0])
        // f = tf.nn.softmax(x)
        // sess.run(tf.compat.v1.global_variables_initializer())

        // sess.run(f)
        assert_eq!(f.eval(), ndarray::arr1(&[0.33333334, 0.33333334, 0.33333334]).into_dyn());

        // sess.run(tf.gradients(f, x))
        assert_eq!(f.gradient("x").eval(), ndarray::arr1(&[0.0, 0.0, 0.0]).into_dyn());
    }
}
