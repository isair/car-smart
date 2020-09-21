const tf = require("@tensorflow/tfjs");

const deserializeModelResult = result => ({
  ...result,
  mean: tf.tensor(result.mean),
  variance: tf.tensor(result.variance),
  weights: tf.tensor(result.weights)
});

module.exports = deserializeModelResult;
