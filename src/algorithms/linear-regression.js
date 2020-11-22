const tf = require('@tensorflow/tfjs');

const gradientDescent = (
  features,
  labels,
  { learningRate = 0.1, weights = tf.ones([features.shape[1], 1]) } = {},
) => {
  const slopes = features
    .transpose()
    .matMul(features.matMul(weights).sub(labels))
    .div(features.shape[0]);
  return weights.sub(slopes.mul(learningRate));
};

const train = (
  features,
  labels,
  {
    learningRate, iterations = 1000, batchSize = 10, initialWeights,
  } = {},
) => {
  let weights = initialWeights;

  let activeLearningRate = learningRate;

  const costHistory = [];
  let lastCost;

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
          weights,
        },
      );
    }

    const cost = features
      .matMul(weights)
      .sub(labels)
      .pow(2)
      .sum()
      .div(features.shape[0])
      .arraySync(); // mean squared error
    costHistory.push(cost);

    if (lastCost && cost) {
      if (cost === lastCost) break;
      activeLearningRate *= cost > lastCost ? 0.5 : 1.05;
    }
    lastCost = cost;
  }

  return { weights, costHistory };
};

const test = (weights, testFeatures, testLabels) => {
  const predictions = testFeatures.matMul(weights);
  const residual = testLabels
    .sub(predictions)
    .pow(2)
    .sum()
    .arraySync();
  const total = testLabels
    .sub(testLabels.mean())
    .pow(2)
    .sum()
    .arraySync();
  return 1 - residual / total;
};

const predict = (observations, weights, mean, variance) => tf
  .ones([observations.shape[0], 1])
  .concat(observations.sub(mean).div(variance.pow(0.5)), 1)
  .matMul(weights);

module.exports = {
  train,
  test,
  predict,
};
