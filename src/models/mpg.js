const tf = require("@tensorflow/tfjs");
const loadCsv = require("tensorflow-load-csv");
const plot = require("node-remote-plot");
const fs = require("fs");
const rimraf = require("rimraf");

const { train, test } = require("../algorithms/linear-regression");

const run = () => {
  let { features, labels, testFeatures, testLabels, mean, variance } = loadCsv(
    "./data/cars.csv",
    {
      featureColumns: ["displacement", "horsepower", "weight"],
      labelColumns: ["mpg"],
      shuffle: true,
      splitTest: 50,
      prependOnes: true,
      standardise: true
    }
  );

  const { weights, mseHistory } = train(features, labels, {
    learningRate: 0.1,
    iterations: 100,
    batchSize: 10
  });

  const accuracy = test(weights, testFeatures, testLabels);

  console.log("mpg r2 is", accuracy);

  plot({
    x: mseHistory,
    xLabel: "Iteration #",
    yLabel: "Mean Squared Error"
  });

  const msePlotImageName = `mpg-mse-${Date.now()}.png`;

  rimraf("src/public/*.png", () => {
    fs.rename("plot.png", `src/public/${msePlotImageName}`, () => {});
  });

  return {
    mean,
    variance,
    weights,
    accuracy,
    plotImageUrl: msePlotImageName
  };
};

module.exports = {
  run
};
