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

  return {
    mean,
    variance,
    weights: tf.tensor([]),
    r2: 0,
    msePlotImageUrl: ""
  };
};

module.exports = {
  run
};
