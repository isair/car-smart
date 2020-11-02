const tf = require("@tensorflow/tfjs");

const gradientDescent = (
  features,
  labels,
  { learningRate = 0.1, weights = tf.ones([features.shape[1], 1]) } = {}
) => {
  const slopes = features
    .transpose()
    .matMul(
      features
        .matMul(weights)
        .sigmoid()
        .sub(labels)
    )
    .div(features.shape[0]);
  return weights.sub(slopes.mul(learningRate));
};

const train = (
  features,
  labels,
  { learningRate, iterations = 1000, batchSize = 10, initialWeights } = {}
) => {
  let weights = initialWeights;

  let activeLearningRate = learningRate;

  const mseHistory = [];
  let lastMse;

  const batchCount = features.shape[0] / batchSize;

  for (let i = 0; i < iterations; i += 1) {
    for (let j = 0; j < batchCount; j += 1) {
      const batchStart = [batchSize * j, 0];
      const currentBatchSize = j === Math.floor(batchCount) ? -1 : batchSize;
      weights = gradientDescent(
        features.slice(batchStart, [currentBatchSize, -1]),
        labels.slice(batchStart, [currentBatchSize, -1]),
        {
          activeLearningRate,
          weights
        }
      );
    }

    const mse = features
      .matMul(weights)
      .sub(labels)
      .pow(2)
      .sum()
      .div(features.shape[0])
      .arraySync();
    mseHistory.push(mse);

    if (lastMse && mse) {
      if (mse === lastMse) break;
      activeLearningRate *= mse > lastMse ? 0.5 : 1.05;
    }
    lastMse = mse;
  }

  return { weights, mseHistory };
};

const predict = (observations, weights, mean, variance) =>
  tf
    .ones([observations.shape[0], 1])
    .concat(observations.sub(mean).div(variance.pow(0.5)), 1)
    .matMul(weights)
    .sigmoid();

const test = (weights, testFeatures, testLabels, mean, variance) => {
  const predictions = testFeatures
    .matMul(weights)
    .sigmoid()
    .round();
  const incorrect = predictions
    .sub(testLabels)
    .abs()
    .sum()
    .arraySync();
  return (predictions.shape[0] - incorrect) / predictions.shape[0];
};

module.exports = {
  train,
  test,
  predict
};
