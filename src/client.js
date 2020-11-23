const tf = require('@tensorflow/tfjs');

const { predict: predictLinear } = require('./algorithms/linear-regression');
const {
  predict: predictLogistic,
} = require('./algorithms/logistic-regression');

const deserializeModelResult = require('./utils/deserializeModelResult');

fetch('/model?name=mpg')
  .then((response) => response.json())
  .then((responseJson) => {
    const {
      mean,
      variance,
      weights,
      accuracy,
      plotImageUrl,
    } = deserializeModelResult(responseJson);

    const accuracyElement = document.querySelector('#mpg-accuracy');
    accuracyElement.innerHTML = `${(accuracy * 100).toFixed(2)}`;

    const mpgPlotElement = document.querySelector('#mpg-cost-plot');
    mpgPlotElement.src = plotImageUrl;

    const inputs = ['#displacement', '#horsepower', '#weight'].map((query) => document.querySelector(query));
    const resultElement = document.querySelector('#mpg-result');

    const calculateMpg = () => {
      const predictions = predictLinear(
        tf.tensor([inputs.map((input) => Number(input.value))]),
        weights,
        mean,
        variance,
      );
      resultElement.innerHTML = `${Number(predictions.arraySync()[0]).toFixed(
        2,
      )} miles per gallon`;
    };

    inputs.forEach((input) => input.addEventListener('change', calculateMpg));
  });

fetch('/model?name=emissions')
  .then((response) => response.json())
  .then((responseJson) => {
    const {
      mean,
      variance,
      weights,
      accuracy,
      plotImageUrl,
    } = deserializeModelResult(responseJson);

    const accuracyElement = document.querySelector('#emissions-accuracy');
    accuracyElement.innerHTML = `${(accuracy * 100).toFixed(2)}`;

    const emissionsPlotElement = document.querySelector('#emissions-cost-plot');
    emissionsPlotElement.src = plotImageUrl;

    const inputs = ['#displacement', '#horsepower', '#weight'].map((query) => document.querySelector(query));
    const resultElement = document.querySelector('#emissions-result');

    const calculateEmissions = () => {
      const predictions = predictLogistic(
        tf.tensor([inputs.map((input) => Number(input.value))]),
        weights,
        mean,
        variance,
      );
      resultElement.innerHTML = Number(predictions.arraySync()[0]) === 1
        ? 'Should pass emissions check'
        : 'Should fail emissions check';
    };

    inputs.forEach((input) => input.addEventListener('change', calculateEmissions));
  });
