const serializeModelResult = (result) => ({
  ...result,
  mean: result.mean.arraySync(),
  variance: result.variance.arraySync(),
  weights: result.weights.arraySync(),
});

module.exports = serializeModelResult;
