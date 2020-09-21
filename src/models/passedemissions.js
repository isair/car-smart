const tf = require("@tensorflow/tfjs");
const loadCsv = require("tensorflow-load-csv");
const plot = require("node-remote-plot");
const fs = require("fs");
const rimraf = require("rimraf");

const { train, test, predict } = require("../algorithms/logistic-regression");

const run = () => {
  let { features, labels, testFeatures, testLabels, mean, variance } = loadCsv(
    "./data/cars.csv",
    {
      featureColumns: ["displacement", "horsepower", "weight"],
      labelColumns: ["passedemissions"],
      mappings: {
        passedemissions: value => (value === "TRUE" ? 1 : 0)
      },
      shuffle: true,
      splitTest: 50,
      prependOnes: true,
      standardise: true
    }
  );

  const { weights } = train(features, labels, {
    learningRate: 0.5,
    iterations: 100,
    batchSize: 50
  });

  const predictions = predict(
    tf.tensor([[97, 88, 1.065]]),
    weights,
    mean,
    variance
  );

  predictions.print();

  return {
    mean,
    variance,
    weights,
    r2: 0,
    plotImageUrl: ""
  };
};

module.exports = {
  run
};
