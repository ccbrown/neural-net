use super::algebra;

pub fn categorical_cross_entropy(prediction: algebra::Expr, truth: algebra::Expr) -> algebra::Expr {
    algebra::expr(1.0) - (truth * prediction.ln()).sum()
}
