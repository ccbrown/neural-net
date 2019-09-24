extern crate flate2;
extern crate ndarray;
extern crate neural_net;

use ndarray::Dimension;

static TEST_IMAGES_GZ: &str = "dataset/test-images.gz";
static TEST_LABELS_GZ: &str = "dataset/test-labels.gz";
static TRAINING_IMAGES_GZ: &str = "dataset/training-images.gz";
static TRAINING_LABELS_GZ: &str = "dataset/training-labels.gz";

static CLASS_NAMES: [&str; 10] = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag",
    "Ankle boot"
];

fn main() -> Result<(), Box<std::error::Error>> {
    neural_net::util::download(neural_net::datasets::FASHION_MNIST_TEST_IMAGES_GZ_URL, TEST_IMAGES_GZ)?;
    neural_net::util::download(neural_net::datasets::FASHION_MNIST_TEST_LABELS_GZ_URL, TEST_LABELS_GZ)?;
    neural_net::util::download(neural_net::datasets::FASHION_MNIST_TRAINING_IMAGES_GZ_URL, TRAINING_IMAGES_GZ)?;
    neural_net::util::download(neural_net::datasets::FASHION_MNIST_TRAINING_LABELS_GZ_URL, TRAINING_LABELS_GZ)?;

    let mut model = neural_net::models::Sequential::new(ndarray::Ix2(28, 28));
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

    let training_images = flate2::read::GzDecoder::new(std::fs::File::open(TRAINING_IMAGES_GZ)?);
    let training_labels = flate2::read::GzDecoder::new(std::fs::File::open(TRAINING_LABELS_GZ)?);
    let mut training_dataset = neural_net::datasets::MNIST::new(training_images, training_labels)?.to_one_hot(CLASS_NAMES.len());

    let mut model = model.compile(training_dataset.target_shape(), neural_net::losses::categorical_cross_entropy);

    model.fit(&mut training_dataset, 5)?;

    Ok(())
}
