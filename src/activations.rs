use super::algebra;

pub fn linear(input: algebra::Expr) -> algebra::Expr {
    input
}

pub fn relu(input: algebra::Expr) -> algebra::Expr {
    input.max(algebra::expr(0.0))
}

pub fn leaky_relu(alpha: f32) -> impl Fn(algebra::Expr) -> algebra::Expr {
    move |input| {
        algebra::ternary(
            algebra::cmp(input.clone(), algebra::cmp::Op::Less, algebra::expr(0.0)),
            input.clone() * algebra::expr(alpha),
            input,
        )
    }
}

pub fn softmax(input: algebra::Expr) -> algebra::Expr {
    input.softmax()
}

#[cfg(test)]
mod tests {
    use super::super::*;
    use super::*;

    #[test]
    fn test_softmax() {
        let x = algebra::v(
            "x",
            Rc::new(algebra::VariableValue::new(ndarray::arr1(&[0.0, 0.0, 0.0]))),
        );
        let f = softmax(x);

        // import tensorflow as tf
        // sess = tf.compat.v1.Session()
        // x = tf.Variable([0.0, 0.0, 0.0])
        // f = tf.nn.softmax(x)
        // sess.run(tf.compat.v1.global_variables_initializer())

        // sess.run(f)
        assert_eq!(
            f.eval(),
            ndarray::arr1(&[0.33333334, 0.33333334, 0.33333334]).into_dyn()
        );

        // sess.run(tf.gradients(f, x))
        assert_eq!(
            f.gradient("x").eval(),
            ndarray::arr1(&[0.0, 0.0, 0.0]).into_dyn()
        );
    }
}
