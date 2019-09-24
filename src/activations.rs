use super::algebra;

pub fn relu(input: algebra::Expr) -> algebra::Expr {
    let zeros = algebra::expr(ndarray::Array::zeros(input.shape()));
    algebra::ternary(
        algebra::cmp(input.clone(), algebra::cmp::Op::Less, zeros.clone()),
        zeros,
        input.clone(),
    )
}

pub fn softmax(input: algebra::Expr) -> algebra::Expr {
    let num = input.exp();
    num.clone() / num.sum()
}
