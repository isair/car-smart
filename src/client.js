const tf = require("@tensorflow/tfjs");

const { predict: predictLinear } = require("./algorithms/linear-regression");

const deserializeModelResult = require("./utils/deserializeModelResult");

fetch("/model?name=mpg")
  .then(response => response.json())
  .then(responseJson => {
    const {
      mean,
      variance,
      weights,
      r2,
      msePlotImageUrl
    } = deserializeModelResult(responseJson);

    const accuracyElement = document.querySelector("#mpg-accuracy");
    accuracyElement.innerHTML = `${(r2 * 100).toFixed(2)}`;

    const msePlotElement = document.querySelector("#mpg-mse-plot");
    msePlotElement.src = msePlotImageUrl;

    const inputs = ["#displacement", "#horsepower", "#weight"].map(query =>
      document.querySelector(query)
    );
    const resultElement = document.querySelector("#result");

    const calculateMpg = () => {
      const predictions = predictLinear(
        tf.tensor([inputs.map(input => Number(input.value))]),
        weights,
        mean,
        variance
      );
      resultElement.innerHTML = `${Number(predictions.arraySync()[0]).toFixed(
        2
      )}`;
    };

    inputs.forEach(input => input.addEventListener("change", calculateMpg));
  });
