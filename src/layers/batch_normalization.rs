use super::super::{algebra, Layer, LayerInstance};
use super::LayerVariablesBuilder;

use ndarray::Dimension;

// Performs batch normalization during inference. This layer only supports inference and does NOT
// implement proper batch normalization during training since this library has no concept of
// batches right now.
pub struct BatchNormalization<
    BetaInitializer,
    GammaInitializer,
    MovingMeanInitializer,
    MovingVarianceInitializer,
> where
    BetaInitializer: Fn(&ndarray::IxDyn) -> ndarray::ArrayD<f32>,
    GammaInitializer: Fn(&ndarray::IxDyn) -> ndarray::ArrayD<f32>,
    MovingMeanInitializer: Fn(&ndarray::IxDyn) -> ndarray::ArrayD<f32>,
    MovingVarianceInitializer: Fn(&ndarray::IxDyn) -> ndarray::ArrayD<f32>,
{
    pub epsilon: f32,
    pub beta_initializer: BetaInitializer,
    pub gamma_initializer: GammaInitializer,
    pub moving_mean_initializer: MovingMeanInitializer,
    pub moving_variance_initializer: MovingVarianceInitializer,
}

impl<BetaInitializer, GammaInitializer, MovingMeanInitializer, MovingVarianceInitializer> Layer
    for BatchNormalization<
        BetaInitializer,
        GammaInitializer,
        MovingMeanInitializer,
        MovingVarianceInitializer,
    >
where
    BetaInitializer: Fn(&ndarray::IxDyn) -> ndarray::ArrayD<f32>,
    GammaInitializer: Fn(&ndarray::IxDyn) -> ndarray::ArrayD<f32>,
    MovingMeanInitializer: Fn(&ndarray::IxDyn) -> ndarray::ArrayD<f32>,
    MovingVarianceInitializer: Fn(&ndarray::IxDyn) -> ndarray::ArrayD<f32>,
{
    fn init(
        self: Box<Self>,
        namespace: &str,
        input_shape: &ndarray::IxDyn,
    ) -> Box<dyn LayerInstance> {
        let mut lv_builder = LayerVariablesBuilder::new(namespace);
        let epsilon = self.epsilon;
        let depth = input_shape.as_array_view()[input_shape.ndim() - 1];
        let beta = lv_builder.append("beta", (self.beta_initializer)(&ndarray::IxDyn(&[depth])));
        let gamma = lv_builder.append("gamma", (self.gamma_initializer)(&ndarray::IxDyn(&[depth])));
        let moving_mean = lv_builder.append(
            "moving_mean",
            (self.moving_mean_initializer)(&ndarray::IxDyn(&[depth])),
        );
        let moving_variance = lv_builder.append(
            "moving_variance",
            (self.moving_variance_initializer)(&ndarray::IxDyn(&[depth])),
        );
        let input_shape = input_shape.clone();

        Box::new(super::Instance {
            expression: move |input| {
                let inv = gamma.clone() / (moving_variance.clone() + epsilon).sqrt();
                input * algebra::broadcast_to(inv.clone(), input_shape.clone())
                    + algebra::broadcast_to(
                        beta.clone() - moving_mean.clone() * inv,
                        input_shape.clone(),
                    )
            },
            variables: lv_builder.variables,
        })
    }
}
