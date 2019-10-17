extern crate byteorder;
#[macro_use] extern crate log;
extern crate env_logger;
extern crate image;
extern crate ndarray;
extern crate neural_net;
#[macro_use] extern crate simple_error;

use std::error::Error;
use std::io::{BufRead, Read};

use byteorder::{LittleEndian, ReadBytesExt};
use ndarray::Dimension;
use neural_net::{activations, algebra, layers};

struct Weights<R: Read> {
    file: R,
}

impl<R: Read> Weights<R> {
    pub fn new(mut file: R) -> Result<Weights<R>, Box<dyn Error>> {
        let major = file.read_u32::<LittleEndian>()?;
        let minor = file.read_u32::<LittleEndian>()?;
        let revision = file.read_u32::<LittleEndian>()?;
        if major != 0 || minor != 2 {
            bail!("unsupported weights file version. got {}.{}.{}, expected 0.2", major, minor, revision);
        }
        file.read_u64::<LittleEndian>()?;
        Ok(Weights{
            file: file,
        })
    }

    pub fn read<D>(&mut self, shape: D) -> Result<ndarray::ArrayD<f32>, Box<dyn Error>> 
        where D: ndarray::Dimension,
    {
        let mut ret = ndarray::Array::zeros(shape.into_dyn());
        self.file.read_f32_into::<LittleEndian>(ret.as_slice_mut().unwrap())?;
        Ok(ret)
    }
}

fn darknet_convolutional<R, A>(weights: &mut Weights<R>, filters: usize, size: usize, stride: usize, batch_normalize: bool, activation: A, channels_in: usize) -> Result<Box<layers::Sequential>, Box<dyn Error>>
    where R: Read,
          A: Fn(algebra::Expr) -> algebra::Expr + 'static,
{
    let vec_shape = ndarray::Ix1(filters).into_dyn();
    let bias = weights.read(vec_shape.clone())?;
    let batch_normalization = match batch_normalize {
        true => Some(layers::BatchNormalization{
                epsilon: 1e-5,
                beta_initializer: neural_net::initializers::copy(bias.clone()),
                gamma_initializer: neural_net::initializers::copy(weights.read(vec_shape.clone())?),
                moving_mean_initializer: neural_net::initializers::copy(weights.read(vec_shape.clone())?),
                moving_variance_initializer: neural_net::initializers::copy(weights.read(vec_shape)?),
            }),
        false => None,
    };
    let mut ret = neural_net::layers::Sequential{
        layers: vec![Box::new(layers::Conv2D{
            activation: activations::linear,
            bias_initializer: neural_net::initializers::copy(bias.clone()),
            kernel_initializer: neural_net::initializers::copy(weights.read(ndarray::Ix4(filters, channels_in, size, size))?.permuted_axes(vec![2, 3, 1, 0])),
            filters: filters,
            kernel_size: ndarray::Ix2(size, size),
            padding: algebra::conv2d::Padding::Same,
            stride: stride,
            use_bias: !batch_normalize,
        })],
    };
    if let Some(l) = batch_normalization {
        ret.layers.push(Box::new(l));
    }
    ret.layers.push(Box::new(layers::Lambda{
        f: activation,
    }));
    Ok(Box::new(ret))
}

fn darknet53_residual<R: Read>(mut weights: &mut Weights<R>, filters1: usize, filters2: usize, channels_in: usize) -> Result<Box<layers::Residual>, Box<dyn Error>> {
    Ok(Box::new(layers::Residual{
        body: Box::new(layers::Sequential{
            layers: vec![
                darknet_convolutional(&mut weights, filters1, 1, 1, true, activations::leaky_relu(0.1), channels_in)?,
                darknet_convolutional(&mut weights, filters2, 3, 1, true, activations::leaky_relu(0.1), filters1)?,
            ],
        }),
    }))
}

fn darknet53<R: Read>(mut weights: Weights<R>) -> Result<layers::Sequential, Box<dyn Error>> {
    Ok(layers::Sequential{
        layers: vec![
            darknet_convolutional(&mut weights, 32, 3, 1, true, activations::leaky_relu(0.1), 3)?,
            darknet_convolutional(&mut weights, 64, 3, 2, true, activations::leaky_relu(0.1), 32)?,
            darknet53_residual(&mut weights, 32, 64, 64)?,
            darknet_convolutional(&mut weights, 128, 3, 2, true, activations::leaky_relu(0.1), 64)?,
            darknet53_residual(&mut weights, 64, 128, 128)?,
            darknet53_residual(&mut weights, 64, 128, 128)?,
            darknet_convolutional(&mut weights, 256, 3, 2, true, activations::leaky_relu(0.1), 128)?,
            darknet53_residual(&mut weights, 128, 256, 256)?,
            darknet53_residual(&mut weights, 128, 256, 256)?,
            darknet53_residual(&mut weights, 128, 256, 256)?,
            darknet53_residual(&mut weights, 128, 256, 256)?,
            darknet53_residual(&mut weights, 128, 256, 256)?,
            darknet53_residual(&mut weights, 128, 256, 256)?,
            darknet53_residual(&mut weights, 128, 256, 256)?,
            darknet53_residual(&mut weights, 128, 256, 256)?,
            darknet_convolutional(&mut weights, 512, 3, 2, true, activations::leaky_relu(0.1), 256)?,
            darknet53_residual(&mut weights, 256, 512, 512)?,
            darknet53_residual(&mut weights, 256, 512, 512)?,
            darknet53_residual(&mut weights, 256, 512, 512)?,
            darknet53_residual(&mut weights, 256, 512, 512)?,
            darknet53_residual(&mut weights, 256, 512, 512)?,
            darknet53_residual(&mut weights, 256, 512, 512)?,
            darknet53_residual(&mut weights, 256, 512, 512)?,
            darknet53_residual(&mut weights, 256, 512, 512)?,
            darknet_convolutional(&mut weights, 1024, 3, 2, true, activations::leaky_relu(0.1), 512)?,
            darknet53_residual(&mut weights, 512, 1024, 1024)?,
            darknet53_residual(&mut weights, 512, 1024, 1024)?,
            darknet53_residual(&mut weights, 512, 1024, 1024)?,
            darknet53_residual(&mut weights, 512, 1024, 1024)?,
            Box::new(layers::GlobalAveragePooling2D{}),
            Box::new(layers::Lambda{f: |x| x.reshape(ndarray::Ix3(1, 1, 1024).into_dyn())}),
            darknet_convolutional(&mut weights, 1000, 1, 1, false, activations::linear, 1024)?,
            Box::new(layers::Lambda{f: |x| x.reshape(ndarray::Ix1(1000).into_dyn())}),
            Box::new(layers::Lambda{f: activations::softmax}),
        ],
    })
}

fn main() -> Result<(), Box<dyn Error>> {
    env_logger::from_env(env_logger::Env::default().default_filter_or("info")).init();

    info!("loading model");
    let weights = Weights::new(std::fs::File::open("darknet53.weights")?)?;
    let mut model = neural_net::models::Sequential::new(ndarray::Ix3(256, 256, 3));
    model.add_layer(darknet53(weights)?)?;

    info!("compiling model");
    let mut model = model.compile_for_inference();

    info!("loading image");
    let image = image::open("dog.jpg")?.to_rgb();
    let shape = ndarray::Ix3(256, 256, 3);
    let image = ndarray::Array::from_shape_fn(shape, |(y, x, c)| image.get_pixel(x as _, y as _)[c] as f32 / 255.0);

    info!("classifying image");
    let prediction = model.predict(image);
    let labels: Vec<_> = std::io::BufReader::new(std::fs::File::open("labels.txt")?).lines().collect();
    let mut indices: Vec<_> = (0..prediction.len()).collect();
    indices.sort_by(|&a, &b| prediction[b].partial_cmp(&prediction[a]).unwrap());
    for &i in indices.iter().take(5) {
        println!("{}: {:.2}%", labels[i].as_ref().unwrap(), prediction[i] * 100.0);
    }

    Ok(())
}
