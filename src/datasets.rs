use std::error::Error;
use std::io::Read;

use byteorder::{BigEndian, ReadBytesExt};
use ndarray::Dimension;

use super::Dataset;

pub static FASHION_MNIST_TRAINING_IMAGES_GZ_URL: &str = "https://github.com/zalandoresearch/fashion-mnist/raw/c624d4501d003356ade2a8a1e6c5055ca9f81dd8/data/fashion/train-images-idx3-ubyte.gz";
pub static FASHION_MNIST_TRAINING_LABELS_GZ_URL: &str = "https://github.com/zalandoresearch/fashion-mnist/raw/c624d4501d003356ade2a8a1e6c5055ca9f81dd8/data/fashion/train-labels-idx1-ubyte.gz";
pub static FASHION_MNIST_TEST_IMAGES_GZ_URL: &str = "https://github.com/zalandoresearch/fashion-mnist/raw/c624d4501d003356ade2a8a1e6c5055ca9f81dd8/data/fashion/t10k-images-idx3-ubyte.gz";
pub static FASHION_MNIST_TEST_LABELS_GZ_URL: &str = "https://github.com/zalandoresearch/fashion-mnist/raw/c624d4501d003356ade2a8a1e6c5055ca9f81dd8/data/fashion/t10k-labels-idx1-ubyte.gz";

pub struct MNISTImageFile<R: Read> {
    file: R,
    images_remaining: usize,
    number_of_rows: usize,
    number_of_columns: usize,
}

impl<R: Read> MNISTImageFile<R> {
    pub fn new(mut file: R) -> Result<MNISTImageFile<R>, Box<dyn Error>> {
        let magic = file.read_u32::<BigEndian>()?;
        if magic != 0x00000803 {
            bail!("invalid magic for mnist image file");
        }
        let number_of_images = file.read_u32::<BigEndian>()?;
        let number_of_rows = file.read_u32::<BigEndian>()?;
        let number_of_columns = file.read_u32::<BigEndian>()?;
        Ok(MNISTImageFile {
            file: file,
            images_remaining: number_of_images as _,
            number_of_rows: number_of_rows as _,
            number_of_columns: number_of_columns as _,
        })
    }
}

impl<R: Read> Iterator for MNISTImageFile<R> {
    type Item = Result<ndarray::Array2<u8>, std::io::Error>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.images_remaining > 0 {
            let mut image = unsafe {
                ndarray::Array::uninitialized((self.number_of_rows, self.number_of_columns))
            };
            if let Err(err) = self.file.read_exact(image.as_slice_mut()?) {
                return Some(Err(err));
            }
            self.images_remaining -= 1;
            Some(Ok(image))
        } else {
            None
        }
    }
}

pub struct MNISTLabelFile<R: Read> {
    bytes: std::io::Bytes<R>,
    labels_remaining: usize,
}

impl<R: Read> MNISTLabelFile<R> {
    pub fn new(mut file: R) -> Result<MNISTLabelFile<R>, Box<dyn Error>> {
        let magic = file.read_u32::<BigEndian>()?;
        if magic != 0x00000801 {
            bail!("invalid magic for mnist label file");
        }
        let number_of_labels = file.read_u32::<BigEndian>()?;
        Ok(MNISTLabelFile {
            bytes: file.bytes(),
            labels_remaining: number_of_labels as _,
        })
    }
}

impl<R: Read> Iterator for MNISTLabelFile<R> {
    type Item = Result<u8, std::io::Error>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.labels_remaining > 0 {
            self.labels_remaining -= 1;
            self.bytes.next()
        } else {
            None
        }
    }
}

pub struct MNIST {
    images: Vec<ndarray::ArrayD<f32>>,
    labels: Vec<ndarray::ArrayD<f32>>,
    target_shape: ndarray::IxDyn,
}

impl MNIST {
    // Creates a new MNIST dataset. The inputs are 2d images with grayscale pixels in the range of
    // 0.0 to 1.0. The targets are zero-dimensional arrays containing the numeric label associated
    // with the input.
    pub fn new<IR: Read, LR: Read>(images: IR, labels: LR) -> Result<MNIST, Box<dyn Error>> {
        let images = MNISTImageFile::new(images)?;
        let mut images: Vec<_> = images.collect();
        for i in 0..images.len() {
            if images[i].is_err() {
                return Err(Box::new(images.remove(i).unwrap_err()));
            }
        }

        let labels = MNISTLabelFile::new(labels)?;
        let mut labels: Vec<_> = labels.collect();
        for i in 0..labels.len() {
            if labels[i].is_err() {
                return Err(Box::new(labels.remove(i).unwrap_err()));
            }
        }

        let images: Vec<_> = images
            .into_iter()
            .map(|x| x.unwrap().mapv(|v| v as f32 / 255.0).into_dyn())
            .collect();
        let labels: Vec<_> = labels
            .into_iter()
            .map(|x| ndarray::arr0(x.unwrap() as f32).into_dyn())
            .collect();

        if images.len() != labels.len() {
            bail!(
                "mismatched image and label counts for mnist dataset ({} and {})",
                images.len(),
                labels.len()
            );
        }

        Ok(MNIST {
            images: images,
            labels: labels,
            target_shape: ndarray::Ix0().into_dyn(),
        })
    }

    // Transforms the targets into one-hot arrays.
    pub fn to_one_hot(self, categories: usize) -> MNIST {
        let labels = self
            .labels
            .into_iter()
            .map(|l| {
                let mut one_hot = ndarray::Array::zeros(categories).into_dyn();
                one_hot[*l.first().unwrap() as usize] = 1.0;
                one_hot
            })
            .collect();
        MNIST {
            images: self.images,
            labels: labels,
            target_shape: ndarray::Ix1(categories).into_dyn(),
        }
    }

    pub fn target_shape(&self) -> ndarray::IxDyn {
        self.target_shape.clone()
    }
}

impl Dataset for MNIST {
    fn len(&self) -> usize {
        self.images.len()
    }

    fn input(&mut self, i: usize) -> Result<ndarray::ArrayViewD<f32>, Box<dyn Error>> {
        Ok(self.images[i].view())
    }

    fn target(&mut self, i: usize) -> Result<ndarray::ArrayViewD<f32>, Box<dyn Error>> {
        Ok(self.labels[i].view())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mnist_image_file() {
        let file: Vec<u8> = vec![
            0, 0, 8, 3, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0, 4, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 1,
            2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2,
        ];
        let mut f = MNISTImageFile::new(&*file).unwrap();
        let image = f.next().unwrap().unwrap();
        assert_eq!(image.rows(), 3);
        assert_eq!(image.cols(), 4);
        assert_eq!(image[(0, 0)], 1);
        assert_eq!(image[(1, 3)], 8);
        let image = f.next().unwrap().unwrap();
        assert_eq!(image[(0, 0)], 1);
        assert_eq!(image[(1, 3)], 8);
        assert_eq!(f.next().is_none(), true);
    }

    #[test]
    fn test_mnist_label_file() {
        let file: Vec<u8> = vec![0, 0, 8, 1, 0, 0, 0, 4, 1, 2, 3, 4];
        let f = MNISTLabelFile::new(&*file).unwrap();
        let mut count = 0;
        for (i, label) in f.enumerate() {
            assert_eq!(label.unwrap(), i as u8 + 1);
            count += 1;
        }
        assert_eq!(count, 4);
    }

    #[test]
    fn test_mnist() {
        let images: Vec<u8> = vec![
            0, 0, 8, 3, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0, 4, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 1,
            2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2,
        ];
        let labels: Vec<u8> = vec![0, 0, 8, 1, 0, 0, 0, 2, 1, 2];
        let mut ds = MNIST::new(&*images, &*labels).unwrap();
        let label1 = ds.target(0).unwrap().into_shape(()).unwrap();
        assert_eq!(label1[()], 1.0);
        assert_eq!(ds.len(), 2);
    }
}
