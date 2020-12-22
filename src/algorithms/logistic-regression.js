const tf = require('@tensorflow/tfjs');

const gradientDescent = (
  features,
  labels,
  { learningRate = 0.1, weights = tf.ones([features.shape[1], labels.shape[1]]) } = {},
) => {
  const slopes = features
    .transpose()
    .matMul(
      features
        .matMul(weights)
        .sigmoid()
        .sub(labels),
    )
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

    const guesses = features.matMul(weights).sigmoid();
    const termOne = labels.transpose().matMul(guesses.log());
    const termTwo = labels.mul(-1).add(1).transpose().matMul(guesses.mul(-1).add(1).log());
    const cost = termOne
      .add(termTwo)
      .div(features.shape[0])
      .mul(-1)
      .arraySync()[0][0]; // cross entropy
    costHistory.push(cost);

    if (lastCost && cost) {
      if (cost === lastCost) break;
      activeLearningRate *= cost > lastCost ? 0.5 : 1.05;
    }
    lastCost = cost;
  }

  return { weights, costHistory };
};

const predict = (
  observations,
  weights,
  mean,
  variance,
  observationsProcessed = false,
  decisionBoundary = 0.5,
) => {
  const observationsToUse = observationsProcessed
    ? observations
    : tf
      .ones([observations.shape[0], 1])
      .concat(observations.sub(mean).div(variance.pow(0.5)), 1);
  return observationsToUse
    .matMul(weights)
    .sigmoid()
    .greater(decisionBoundary)
    .cast('float32');
};

const test = (
  weights,
  testFeatures,
  testLabels,
  mean,
  variance,
  testFeaturesProcessed = true,
  decisionBoundary = 0.5,
) => {
  const predictions = predict(
    testFeatures,
    weights,
    mean,
    variance,
    testFeaturesProcessed,
    decisionBoundary,
  );
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
  predict,
};
