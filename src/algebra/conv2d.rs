use std::fmt;

use super::{Expr, ExprImpl};

use ndarray::Dimension;

#[derive(Clone, Copy)]
pub enum Padding {
    Same,
    Valid,
}

impl fmt::Display for Padding {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", match self {
            Padding::Same => "SAME",
            Padding::Valid => "VALID",
        })
    }
}

// Conv2D performs a 2-dimensional convolution. The input is expected to be of shape (in_height,
// in_width, in_channels). The kernel is expected to be of shape (kernel_height, kernel_width,
// in_channels, out_channels).
pub struct Conv2D {
    pub input: Expr,
    pub kernel: Expr,
    pub stride: usize,
    pub padding: Padding,
}

impl ExprImpl for Conv2D {
    fn eval_inputs(&self, inputs: &Vec<ndarray::ArrayD<f32>>) -> ndarray::ArrayD<f32> {
        let (input, kernel) = (&inputs[0], &inputs[1]);
        let kernel_shape = kernel.shape();
        let (kernel_height, kernel_width, in_channels, out_channels) = (kernel_shape[0], kernel_shape[1], kernel_shape[2], kernel_shape[3]);

        // If we're using "same" padding, replace the input with a zero-padded image.
        let input = match self.padding {
            Padding::Same => {
                let (y_padding, x_padding) = ((kernel_height - 1) / 2, (kernel_width - 1) / 2);
                let in_shape = input.shape();
                let (in_height, in_width) = (in_shape[0], in_shape[1]);
                // Even kernel dimensions are rare, but to match Tensorflow, we'll add an extra zero to the end of each such dimension.
                let (padded_height, padded_width) = ((in_height + 2 * y_padding + (kernel_height + 1) % 2), (in_width + 2 * x_padding + (kernel_width + 1) % 2));
                let mut padded = ndarray::Array::zeros((padded_height, padded_width, in_channels));
                padded.slice_mut(s![y_padding..y_padding+in_height, x_padding..x_padding+in_width, ..]).assign(&input);
                padded.into_dyn()
            },
            _ => input.clone(),
        };

        let in_shape = input.shape();
        let (in_height, in_width) = (in_shape[0], in_shape[1]);

        // Flatten the kernel so the convolution can be performed as a series of matrix-vector multiplications.
        let mut flattened_kernel = ndarray::Array::zeros((out_channels, kernel_height * kernel_width * in_channels));
        let mut temp_patch = ndarray::Array::zeros((kernel_height, kernel_width, in_channels));
        for c in 0..out_channels {
            temp_patch.assign(&kernel.slice(s![.., .., .., c]));
            flattened_kernel.slice_mut(s![c, ..]).assign(&temp_patch.clone().into_shape(kernel_height * kernel_width * in_channels).unwrap());
        }

        let mut out = ndarray::Array::zeros(((in_height - kernel_height) / self.stride + 1, (in_width - kernel_width) / self.stride + 1, out_channels));
        for y in 0..out.shape()[0] {
            for x in 0..out.shape()[1] {
                let (y_min, x_min) = (y * self.stride, x * self.stride);
                let (y_max, x_max) = (y_min + kernel_height, x_min + kernel_width);
                temp_patch.assign(&input.slice(s![y_min..y_max, x_min..x_max, ..]));
                let patch = temp_patch.clone().into_shape(kernel_height * kernel_width * in_channels).unwrap();
                ndarray::linalg::general_mat_vec_mul(1.0, &flattened_kernel, &patch, 0.0, &mut out.slice_mut(s![y, x, ..]));
            }
        }
        out.into_dyn()
    }

    fn shape(&self) -> ndarray::IxDyn {
        let in_shape = self.input.shape();
        let in_height = in_shape.as_array_view()[0];
        let in_width = in_shape.as_array_view()[1];
        let kernel_shape = self.kernel.shape();
        let out_channels = kernel_shape.as_array_view()[3];
        match self.padding {
            Padding::Same => ndarray::Ix3(in_height / self.stride, in_width / self.stride, out_channels),
            Padding::Valid => {
                let kernel_height = kernel_shape.as_array_view()[0];
                let kernel_width = kernel_shape.as_array_view()[1];
                ndarray::Ix3((in_height - kernel_height) / self.stride + 1, (in_width - kernel_width) / self.stride + 1, out_channels)
            },
        }.into_dyn()
    }

    fn is_constant(&self) -> bool {
        self.input.is_constant() && self.kernel.is_constant()
    }

    fn propagate_constants(&self) -> Expr {
        if self.is_constant() {
            super::expr(self.eval())
        } else {
            conv2d(self.input.propagate_constants(), self.kernel.propagate_constants(), self.stride, self.padding)
        }
    }

    fn accumulate_gradients(&self, _output: Expr, gradients: &mut super::Gradients) {
        // None of the examples require training of neural nets with convolutional layers, so I'll just
        // leave this as an "exercise for the reader".
        let reason = "Conv2D gradients have not been implemented.";
        self.input.accumulate_gradients(super::unevaluable(self.input.shape().clone(), reason), gradients);
        self.kernel.accumulate_gradients(super::unevaluable(self.kernel.shape().clone(), reason), gradients);
    }

    fn inputs(&self) -> Vec<&Expr> {
        vec![&self.input, &self.kernel]
    }
}

impl fmt::Display for Conv2D {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "conv2d({}, {}, {}, {})", self.input, self.kernel, self.stride, self.padding)
    }
}

pub fn conv2d<A: Into<Expr>, B: Into<Expr>>(input: A, kernel: B, stride: usize, padding: Padding) -> Expr {
    Expr::new(Conv2D{
        input: input.into(),
        kernel: kernel.into(),
        stride: stride,
        padding: padding,
    })
}

#[cfg(test)]
mod tests {
    use super::super::*;

    #[test]
    fn test() {
        let img = expr(ndarray::arr3(&[[[0.0, 1.0], [1.0, 2.0]], [[2.0, 3.0], [3.0, 4.0]]]));
        let mut kernel = ndarray::Array::zeros((1, 1, 2, 3));
        kernel.slice_mut(s![0, .., .., ..]).assign(&ndarray::arr3(&[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]));
        assert_eq!(conv2d(img.clone(), kernel.clone(), 1, Padding::Valid).eval(), ndarray::arr3(&[[[4., 5., 6.], [9., 12., 15.]], [[14., 19., 24.], [19., 26., 33.]]]).into_dyn());
        assert_eq!(conv2d(img.clone(), kernel.clone(), 2, Padding::Valid).eval(), ndarray::arr3(&[[[4., 5., 6.]]]).into_dyn());

        let mut kernel = ndarray::Array::zeros((1, 2, 2, 3));
        kernel.slice_mut(s![0, .., .., ..]).assign(&ndarray::arr3(&[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[11.0, 12.0, 13.0], [14.0, 15.0, 16.0]]]));
        assert_eq!(conv2d(img.clone(), kernel.clone(), 1, Padding::Valid).eval(), ndarray::arr3(&[[[43., 47., 51.]], [[103., 115., 127.]]]).into_dyn());
        assert_eq!(conv2d(img.clone(), kernel.clone(), 1, Padding::Same).eval(), ndarray::arr3(&[[[43., 47., 51.], [9., 12., 15.]], [[103., 115., 127.], [19., 26., 33.]]]).into_dyn());
    }
}
