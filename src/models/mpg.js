require("@tensorflow/tfjs-node");

const tf = require("@tensorflow/tfjs");
const loadCsv = require("tensorflow-load-csv");
const plot = require("node-remote-plot");
const fs = require("fs");
const rimraf = require("rimraf");

const {
  train: trainLinear,
  test: testLinear
} = require("../algorithms/linear-regression");

const run = () => {
  let { features, labels, testFeatures, testLabels, mean, variance } = loadCsv(
    "./data/cars.csv",
    {
      featureColumns: ["horsepower", "weight", "displacement"],
      labelColumns: ["mpg"],
      shuffle: true,
      splitTest: 50,
      prependOnes: true,
      standardise: true
    }
  );

  const mpg = trainLinear(features, labels, {
    learningRate: 0.1,
    iterations: 100,
    batchSize: 10
  });

  const r2 = testLinear(mpg.weights, testFeatures, testLabels);

  console.log("mpg r2 is", r2);

  const msePlotImageName = `mpg-mse-${Date.now()}.png`;

  plot({
    x: mpg.mseHistory,
    xLabel: "Iteration #",
    yLabel: "Mean Squared Error"
  });

  rimraf("src/public/*.png", () => {
    fs.rename("plot.png", `src/public/${msePlotImageName}`, () => {});
  });

  return {
    mean,
    variance,
    weights: mpg.weights,
    r2: mpg.r2,
    msePlotImageUrl: msePlotImageName
  };
};

module.exports = {
  run
};
