extern crate reqwest;
#[macro_use] extern crate simple_error;

use std::error::Error;

pub static FASHION_MNIST_TRAINING_IMAGES_URL: &str = "https://github.com/zalandoresearch/fashion-mnist/raw/c624d4501d003356ade2a8a1e6c5055ca9f81dd8/data/fashion/train-images-idx3-ubyte.gz";
pub static FASHION_MNIST_TRAINING_LABELS_URL: &str = "https://github.com/zalandoresearch/fashion-mnist/raw/c624d4501d003356ade2a8a1e6c5055ca9f81dd8/data/fashion/train-labels-idx1-ubyte.gz";
pub static FASHION_MNIST_TEST_IMAGES_URL: &str = "https://github.com/zalandoresearch/fashion-mnist/raw/c624d4501d003356ade2a8a1e6c5055ca9f81dd8/data/fashion/t10k-images-idx3-ubyte.gz";
pub static FASHION_MNIST_TEST_LABELS_URL: &str = "https://github.com/zalandoresearch/fashion-mnist/raw/c624d4501d003356ade2a8a1e6c5055ca9f81dd8/data/fashion/t10k-labels-idx1-ubyte.gz";

pub fn download(url: &str, destination: &str) -> Result<(), Box<Error>> {
    let path = std::path::Path::new(destination);
    if path.exists() {
        return Ok(());
    }
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut resp = reqwest::get(url)?;
    if !resp.status().is_success() {
        bail!("unexpected status code: {}", resp.status());
    }
    let mut file = std::fs::File::create(path)?;
    std::io::copy(&mut resp, &mut file)?;
    Ok(())
}
