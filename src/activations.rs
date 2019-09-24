use super::algebra;

pub fn relu(input: ndarray::ArrayD<algebra::Expr>) -> ndarray::ArrayD<algebra::Expr> {
    input.mapv(|e| algebra::ternary(
        algebra::cmp(e.clone(), algebra::cmp::Op::Less, algebra::c(0.0)),
        algebra::c(0.0),
        e.clone(),
    ))
}

pub fn softmax(input: ndarray::ArrayD<algebra::Expr>) -> ndarray::ArrayD<algebra::Expr> {
    let num = input.mapv_into(|x| x.exp());
    let den = num.fold(algebra::c(0.0), |sum, e| sum + e.clone());
    num.mapv(|num| num / den.clone())
}
