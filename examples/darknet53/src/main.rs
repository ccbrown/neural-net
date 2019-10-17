extern crate byteorder;
#[macro_use] extern crate log;
extern crate env_logger;
extern crate ndarray;
extern crate neural_net;
#[macro_use] extern crate simple_error;

use std::error::Error;
use std::io::Read;

use byteorder::{LittleEndian, ReadBytesExt};
use ndarray::Dimension;
use neural_net::{activations, algebra, layers};

struct Weights<R: Read> {
    _file: R,
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
            _file: file,
        })
    }
}

fn darknet_convolutional<R, A>(_weights: &Weights<R>, filters: usize, size: usize, stride: usize, batch_normalize: bool, activation: A) -> Box<layers::Sequential>
    where R: Read,
          A: Fn(algebra::Expr) -> algebra::Expr + 'static,
{
    let mut ret = neural_net::layers::Sequential{
        layers: vec![Box::new(layers::Conv2D{
            activation: activations::linear,
            kernel_initializer: neural_net::initializers::zeros, // TODO
            filters: filters,
            kernel_size: ndarray::Ix2(size, size),
            padding: algebra::conv2d::Padding::Same,
            stride: stride,
            use_bias: !batch_normalize,
        })],
    };
    if batch_normalize {
        ret.layers.push(Box::new(layers::BatchNormalization{
            epsilon: 1e-5,
            beta_initializer: neural_net::initializers::zeros, // TODO
            gamma_initializer: neural_net::initializers::zeros, // TODO
            moving_mean_initializer: neural_net::initializers::zeros, // TODO
            moving_variance_initializer: neural_net::initializers::zeros, // TODO
        }));
    }
    ret.layers.push(Box::new(layers::Lambda{
        f: activation,
    }));
    Box::new(ret)
}

fn darknet53_residual<R: Read>(weights: &Weights<R>, filters1: usize, filters2: usize) -> Box<layers::Residual> {
    Box::new(layers::Residual{
        body: Box::new(layers::Sequential{
            layers: vec![
                darknet_convolutional(&weights, filters1, 1, 1, true, activations::leaky_relu(0.1)),
                darknet_convolutional(&weights, filters2, 3, 1, true, activations::leaky_relu(0.1)),
            ],
        }),
    })
}

fn darknet53<R: Read>(weights: Weights<R>) -> Result<layers::Sequential, Box<dyn Error>> {
    Ok(layers::Sequential{
        layers: vec![
            darknet_convolutional(&weights, 32, 3, 1, true, activations::leaky_relu(0.1)),
            darknet_convolutional(&weights, 64, 3, 2, true, activations::leaky_relu(0.1)),
            darknet53_residual(&weights, 32, 64),
            darknet_convolutional(&weights, 128, 3, 2, true, activations::leaky_relu(0.1)),
            darknet53_residual(&weights, 64, 128),
            darknet53_residual(&weights, 64, 128),
            darknet_convolutional(&weights, 256, 3, 2, true, activations::leaky_relu(0.1)),
            darknet53_residual(&weights, 128, 256),
            darknet53_residual(&weights, 128, 256),
            darknet53_residual(&weights, 128, 256),
            darknet53_residual(&weights, 128, 256),
            darknet53_residual(&weights, 128, 256),
            darknet53_residual(&weights, 128, 256),
            darknet53_residual(&weights, 128, 256),
            darknet53_residual(&weights, 128, 256),
            darknet_convolutional(&weights, 512, 3, 2, true, activations::leaky_relu(0.1)),
            darknet53_residual(&weights, 256, 512),
            darknet53_residual(&weights, 256, 512),
            darknet53_residual(&weights, 256, 512),
            darknet53_residual(&weights, 256, 512),
            darknet53_residual(&weights, 256, 512),
            darknet53_residual(&weights, 256, 512),
            darknet53_residual(&weights, 256, 512),
            darknet53_residual(&weights, 256, 512),
            darknet_convolutional(&weights, 1024, 3, 2, true, activations::leaky_relu(0.1)),
            darknet53_residual(&weights, 512, 1024),
            darknet53_residual(&weights, 512, 1024),
            darknet53_residual(&weights, 512, 1024),
            darknet53_residual(&weights, 512, 1024),
            // keras.layers.GlobalAveragePooling2D(),
            Box::new(layers::Lambda{f: |x| x.reshape(ndarray::Ix3(1, 1, 1024).into_dyn())}),
            darknet_convolutional(&weights, 1000, 1, 1, false, activations::linear),
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
    // TODO

    info!("classifying image");
    model.predict(ndarray::Array::zeros(ndarray::Ix3(256, 256, 3)));

    Ok(())
}
