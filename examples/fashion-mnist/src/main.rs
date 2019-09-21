extern crate neural_net;

use std::error::Error;

fn main() -> Result<(), Box<Error>> {
    static TEST_IMAGES: &str = "dataset/test-images.gz";
    static TEST_LABELS: &str = "dataset/test-labels.gz";
    static TRAINING_IMAGES: &str = "dataset/training-images.gz";
    static TRAINING_LABELS: &str = "dataset/training-labels.gz";

    neural_net::download(neural_net::FASHION_MNIST_TEST_IMAGES_URL, TEST_IMAGES)?;
    neural_net::download(neural_net::FASHION_MNIST_TEST_LABELS_URL, TEST_LABELS)?;
    neural_net::download(neural_net::FASHION_MNIST_TRAINING_IMAGES_URL, TRAINING_IMAGES)?;
    neural_net::download(neural_net::FASHION_MNIST_TRAINING_LABELS_URL, TRAINING_LABELS)?;

    Ok(())
}
