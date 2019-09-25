use super::algebra;

pub fn relu(input: algebra::Expr) -> algebra::Expr {
    algebra::ternary(
        algebra::cmp(input.clone(), algebra::cmp::Op::Less, algebra::expr(0.0)),
        algebra::expr(0.0),
        input.clone(),
    )
}

pub fn softmax(input: algebra::Expr) -> algebra::Expr {
    let num = input.exp();
    num.clone() / num.sum()
}
