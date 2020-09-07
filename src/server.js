require("@tensorflow/tfjs-node");

const express = require("express");
const tf = require("@tensorflow/tfjs");
const loadCsv = require("tensorflow-load-csv");
const plot = require("node-remote-plot");
const fs = require("fs");
const rimraf = require("rimraf");

const { train: trainLinear, test: testLinear } = require("./linear-regression");

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

// MPG model - linear regression

const { weights, mseHistory } = trainLinear(features, labels, {
  learningRate: 0.1,
  iterations: 100,
  batchSize: 10
});

const r2 = testLinear(weights, testFeatures, testLabels);

console.log("mpg r2 is", r2);

const timestamp = Date.now();

plot({
  x: mseHistory,
  xLabel: "Iteration #",
  yLabel: "Mean Squared Error"
});

rimraf("src/public/*.png", () => {
  fs.rename("plot.png", `src/public/mpg-mse-${timestamp}.png`, () => {});
});

// Web server logic

const app = express();

app.use(express.static(__dirname + "/public"));

app.get("/", (request, response) => {
  response.sendFile(__dirname + "/views/index.html");
});

app.get("/model", (request, response) => {
  response.json({
    timestamp,
    weights: weights.arraySync(),
    mean: mean.arraySync(),
    variance: variance.arraySync(),
    r2
  });
});

const listener = app.listen(process.env.PORT, () => {
  console.log("Your app is listening on port " + listener.address().port);
});
