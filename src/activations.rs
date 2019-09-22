pub fn relu(input: ndarray::ArrayD<f32>) -> ndarray::ArrayD<f32> {
    input.mapv_into(|x| x.max(0.0))
}

pub fn softmax(input: ndarray::ArrayD<f32>) -> ndarray::ArrayD<f32> {
    let num = input.mapv_into(|x| x.exp());
    let den = num.sum();
    num / den
}
