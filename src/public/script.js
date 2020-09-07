fetch("/model")
  .then(response => response.json())
  .then(responseJson => {
    const { timestamp, weights, mean, variance, r2 } = responseJson;

    const accuracyElement = document.querySelector("#mpg-accuracy");
    accuracyElement.innerHTML = `${(r2 * 100).toFixed(2)}`;

    const msePlotElement = document.querySelector("#mpg-mse-plot");
    msePlotElement.src = `mpg-mse-${timestamp}.png`;

    const inputs = ["#displacement", "#horsepower", "#weight"].map(query =>
      document.querySelector(query)
    );
    const resultElement = document.querySelector("#result");

    const calculateMpg = () => {
      let mpg = weights.reduce(
        (acc, weight, i) =>
          acc +
          weight *
            (i === 0
              ? 1
              : ((Number(inputs[i - 1].value) || 0) - mean[i - 1]) /
                Math.sqrt(variance[i - 1])),
        0
      );
      resultElement.innerHTML = `${mpg.toFixed(2)}`;
    };

    inputs.forEach(input => input.addEventListener("change", calculateMpg));
  });
