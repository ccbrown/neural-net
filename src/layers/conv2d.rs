use super::{LayerVariablesBuilder};
use super::super::{algebra, Layer, LayerInstance};

use ndarray::Dimension;

use algebra::conv2d::Padding;

// Conv2D takes a 1-dimensional input and emits a 1-dimensional output.
pub struct Conv2D<Activation, KernelInitializer>
    where Activation: Fn(algebra::Expr) -> algebra::Expr + 'static,
          KernelInitializer: Fn(&ndarray::IxDyn) -> ndarray::ArrayD<f32>
{
    pub activation: Activation,
    pub kernel_initializer: KernelInitializer,
    pub filters: usize,
    pub kernel_size: ndarray::Ix2,
    pub padding: Padding,
    pub stride: usize,
    pub use_bias: bool,
}

impl<Activation, KernelInitializer> Layer for Conv2D<Activation, KernelInitializer>
    where Activation: Fn(algebra::Expr) -> algebra::Expr + Clone + 'static,
          KernelInitializer: Fn(&ndarray::IxDyn) -> ndarray::ArrayD<f32>
{
    fn init(&self, namespace: &str, input_shape: &ndarray::IxDyn) -> Box<LayerInstance> {
        let mut lv_builder = LayerVariablesBuilder::new(namespace);
        let in_channels = input_shape.as_array_view()[2];
        let biases = match self.use_bias {
            true => Some(lv_builder.append("b", ndarray::Array::zeros(self.filters))),
            false => None,
        };
        let kernel = lv_builder.append("w", (self.kernel_initializer)(&ndarray::IxDyn(&[self.kernel_size[0], self.kernel_size[1], in_channels, self.filters])));
        let activation = self.activation.clone();
        let stride = self.stride;
        let padding = self.padding;
        Box::new(super::Instance{
            expression: move |input| {
                let mut result = algebra::conv2d(input, kernel.clone(), stride, padding);
                if let Some(ref biases) = biases {
                    result = algebra::bias_add(result, biases.clone());
                }
                (activation)(result)
            },
            variables: lv_builder.variables,
        })
    }
}
