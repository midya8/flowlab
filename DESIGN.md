# Direction, gradients and bounds

## Data-to-latent convention

Forward means f:x->z and contributes +sum(s) to log p_X(x). Inverse means z->x and contributes -sum(s) evaluated at inverse conditioning coordinates. Layers alternate a=layer%2, b=1-a. Neither permutation nor any hidden extra normalization is applied. Conditioner hidden activations are tanh(w_j*x_a+b_j); two separate linear output heads share those activations. Scale is B*tanh(raw_s); translation is raw_t. This differs from the paper's image-scale architecture and is explicitly a small educational variant.

Parameters per layer are input weights H, hidden biases H, scale-output weights H, scale-output bias1, translation-output weights H, translation-output bias1, in that order. Thus 4H+2 per layer; default4*(4*12+2)=200. Initial output weights are small Gaussian draws, not exactly zero; the untrained control uses identical initialization. B is fixed in the artifact, not learned.

## Exact likelihood derivative

NLL = log(2pi)+||z||^2/2 - sum(s). Start dZ=z. For a layer with y_b=x_b*exp(s)+t:

- dS = dY_b*x_b*exp(s) - 1. The -1 comes from this layer's log determinant.
- dT = dY_b.
- dRawS = dS * B * (1-tanh(raw_s)^2).
- Backpropagate the shared tanh conditioner into its parameters and conditioning input x_a.
- dX_a = dY_a + conditioner input derivative; dX_b=dY_b*exp(s).

The reverse loop retains each original x_b and conditioner activations; omitting the logdet term changes the objective and is caught by finite differences/sign tests. Gradients are for NLL per 2-D point, averaged over minibatch. Adam uses beta1=.9, beta2=.999, epsilon1e-8 with bias correction. No decay, clipping, EMA, learning-rate schedule, early stopping or selected checkpoint. Training history records stochastic minibatch losses before updates, not monotone fixed-set validation losses.

## Supported inputs and artifacts

All points are finite numeric length-2 lists; booleans fail. Training uses 2-8192 rows, coordinates[-100,100], updates1-10000, batch1-128, learning rate[1e-6,.1], seed integer[0,2^32-1]. Architecture: layers1-8, hidden1-32, fixed scale bound[.01,2]. Sample count1-2048; CLI score count1-8192. Bounds constrain resources and shape, not convergence or global numeric safety. Scoring/inversion accepts finite coordinates outside the training range, but overflow/nonfinite results fail. Extreme valid artifact parameters can still fail numerically; finite schema validation is not a conditioning certificate.

`flowlab_v1` exact fields: format/layers/hidden/scale_bound/parameters. Missing/extra keys, unknown versions, wrong parameter counts, bools/nonfinite values and invalid architecture fail. No training data, Gaussian baseline or optimizer state is included. JSON writes prohibit NaN. CLI error exits2 without a traceback. Train/score/sample do not mutate input rows or an inference model.

## Fixture and baselines

The banana fixture is constructed directly from independent normal u,v: x=u,y=.5*v+.6*(u^2-1). Its inverse is u=x,v=2*(y-.6*(x^2-1)); |det(d(u,v)/d(x,y))|=2. Thus reference logdensity is standard2Dnormal(u,v)-log(.5). The reference is report-only, not consulted by flow training. It does appear in sample diagnostics after training, so the result is development evidence, not a hidden test benchmark.

The full-covariance Gaussian uses training ML mean and covariance denominatorN, no fixed diagonal, no use of test rows and no hyperparameter selection. Its inverse2x2 formula is checked independently. The fixture covariance is nonsingular, so the demo does not claim a general-purpose covariance regularizer. Flow/untrained/Gaussian sampling shares the same standard-normal seed/draw sequence; the generator reference has an independent seed. Likelihood and sample summaries are descriptive; no confidence interval, multi-seed claim, causal attribution or anomaly threshold is supplied.

A bijection of a normalized normal base defines a normalized model density mathematically. Midpoint quadrature on one small near-identity model checks implementation on a finite square, not tail mass generally. Roundtrip residuals and Jacobian differences are measured audits, not certified bounds. Larger likelihood or smaller reconstruction error does not alone imply correct tail behavior or sample quality.

## Input score decomposition

For f:x->z and L=sum(s), gradient_x log p_X is J_f^T(-z)+gradient_x L. The implementation independently reverses both components through each cached coupling. Base adjoint starts -z, volume adjoint starts0. For each component, dS=dY_b*x_b*exp(s)+c with c=0 for base and c=1 for volume; dT=dY_b. Both propagate through the shared conditioner and unchanged coordinate. The volume component includes downstream layers as well as the current layer's direct +1. The returned full vector is the coordinate-wise sum; no parameter gradients, weight updates or clipping occur.

The scalar-density command is named score for backward compatibility; the new gradient command is unambiguously a vector and limited to1-256 rows. All normal point/schema checks apply; nonfinite output in any component fails. Separate finite-difference checks of log density, normal-base energy and logdet plus closed-form cases avoid testing the score only against its own decomposition. A fixed grid checks the trained model, but says nothing about maximum error elsewhere. No automatic step size, score-based sampler, mode finder, OOD threshold or explanation of causal influence is claimed. Nonlinear reparameterization changes both density and its gradient; even unit scaling changes the reported vector.
