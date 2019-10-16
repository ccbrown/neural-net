use super::{LayerVariablesBuilder};
use super::super::{algebra, Layer, LayerInstance, LayerVariable};

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
        Box::new(Conv2DInstance{
            activation: self.activation.clone(),
            biases: match self.use_bias {
                true => Some(lv_builder.append("b", ndarray::Array::zeros(self.filters))),
                false => None,
            },
            kernel: lv_builder.append("w", (self.kernel_initializer)(&ndarray::IxDyn(&[self.kernel_size[0], self.kernel_size[1], in_channels, self.filters]))),
            padding: self.padding,
            stride: self.stride,
            variables: lv_builder.variables,
        })
    }

    fn output_shape(&self, input_shape: &ndarray::IxDyn) -> ndarray::IxDyn {
        let in_height = input_shape.as_array_view()[0];
        let in_width = input_shape.as_array_view()[1];
        match self.padding {
            Padding::Same => ndarray::Ix3(in_height / self.stride, in_width / self.stride, self.filters),
            Padding::Valid => {
                let kernel_height = self.kernel_size[0];
                let kernel_width = self.kernel_size[1];
                ndarray::Ix3((in_height - kernel_height) / self.stride + 1, (in_width - kernel_width) / self.stride + 1, self.filters)
            },
        }.into_dyn()
    }
}

pub struct Conv2DInstance<Activation>
    where Activation: Fn(algebra::Expr) -> algebra::Expr
{
    activation: Activation,
    biases: Option<algebra::Expr>,
    kernel: algebra::Expr,
    padding: Padding,
    stride: usize,
    variables: Vec<LayerVariable>,
}

impl<Activation> LayerInstance for Conv2DInstance<Activation>
    where Activation: Fn(algebra::Expr) -> algebra::Expr
{
    fn expression(&self, input: algebra::Expr) -> algebra::Expr {
        let mut result = algebra::conv2d(input, self.kernel.clone(), self.stride, self.padding);
        if let Some(ref biases) = self.biases {
            result = algebra::bias_add(result, biases.clone());
        }
        (self.activation)(result)
    }

    fn variables(&self) -> &[LayerVariable] {
        self.variables.as_slice()
    }
}
