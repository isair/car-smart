const loadCsv = require('tensorflow-load-csv');
const plot = require('node-remote-plot');
const rimraf = require('rimraf');

const { train, test } = require('../algorithms/logistic-regression');

const run = () => {
  const {
    features, labels, testFeatures, testLabels, mean, variance,
  } = loadCsv(
    './data/cars.csv',
    {
      featureColumns: ['displacement', 'horsepower', 'weight'],
      labelColumns: ['passedemissions'],
      mappings: {
        passedemissions: (value) => (value === 'TRUE' ? 1 : 0),
      },
      shuffle: true,
      splitTest: 50,
      prependOnes: true,
      standardise: true,
    },
  );

  const { weights, costHistory } = train(features, labels, {
    learningRate: 0.5,
    iterations: 100,
    batchSize: 50,
  });

  const accuracy = test(weights, testFeatures, testLabels, mean, variance);

  console.log('passedemissions accuracy is', accuracy);

  const plotImageName = `passedemissions/cost-${Date.now()}`;

  rimraf('src/public/passedemissions/*.png', () => {
    plot({
      x: costHistory,
      xLabel: 'Iteration #',
      yLabel: 'Cross Entropy',
      name: `src/public/${plotImageName}`,
    });
  });

  return {
    mean,
    variance,
    weights,
    accuracy,
    plotImageUrl: `${plotImageName}.png`,
  };
};

module.exports = {
  run,
};
