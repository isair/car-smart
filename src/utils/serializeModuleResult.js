const serializeModelResult = result => ({
  mean: result.mean.arraySync(),
  variance: result.variance.arraySync(),
  weights: result.weights.arraySync(),
  r2: result.r2,
  msePlotImageUrl: result.msePlotImageUrl
});

module.exports = serializeModelResult;
