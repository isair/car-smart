const loadCsv = require('tensorflow-load-csv');
const plot = require('node-remote-plot');
const rimraf = require('rimraf');

const { train, test } = require('../algorithms/linear-regression');

const run = () => {
  const {
    features, labels, testFeatures, testLabels, mean, variance,
  } = loadCsv(
    './data/cars.csv',
    {
      featureColumns: ['displacement', 'horsepower', 'weight'],
      labelColumns: ['mpg'],
      shuffle: true,
      splitTest: 50,
      prependOnes: true,
      standardise: true,
    },
  );

  const { weights, costHistory } = train(features, labels, {
    learningRate: 0.1,
    iterations: 100,
    batchSize: 10,
  });

  const accuracy = test(weights, testFeatures, testLabels);

  console.log('mpg r2 is', accuracy);

  const plotImageName = `mpg/cost-${Date.now()}`;

  rimraf('src/public/mpg/*.png', () => {
    plot({
      x: costHistory,
      xLabel: 'Iteration #',
      yLabel: 'Mean Squared Error',
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
