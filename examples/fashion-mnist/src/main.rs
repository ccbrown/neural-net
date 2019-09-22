extern crate neural_net;

fn main() -> Result<(), Box<std::error::Error>> {
    static TEST_IMAGES: &str = "dataset/test-images.gz";
    static TEST_LABELS: &str = "dataset/test-labels.gz";
    static TRAINING_IMAGES: &str = "dataset/training-images.gz";
    static TRAINING_LABELS: &str = "dataset/training-labels.gz";

    neural_net::util::download(neural_net::FASHION_MNIST_TEST_IMAGES_URL, TEST_IMAGES)?;
    neural_net::util::download(neural_net::FASHION_MNIST_TEST_LABELS_URL, TEST_LABELS)?;
    neural_net::util::download(neural_net::FASHION_MNIST_TRAINING_IMAGES_URL, TRAINING_IMAGES)?;
    neural_net::util::download(neural_net::FASHION_MNIST_TRAINING_LABELS_URL, TRAINING_LABELS)?;

    let mut model = neural_net::models::Sequential::new(neural_net::Shape::D2(28, 28));
    model.add_layer(neural_net::layers::Flatten{
        input_shape: model.output_shape(),
    })?;
    model.add_layer(neural_net::layers::Dense{
        input_size: model.output_shape().size(),
        output_size: 128,
        activation: neural_net::activations::relu,
        kernel_initializer: neural_net::initializers::glorot_uniform,
    })?;
    model.add_layer(neural_net::layers::Dense{
        input_size: model.output_shape().size(),
        output_size: 10,
        activation: neural_net::activations::softmax,
        kernel_initializer: neural_net::initializers::glorot_uniform,
    })?;

    let _model = model.compile();
    // TODO

    Ok(())
}
