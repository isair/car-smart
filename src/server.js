const express = require("express");
const mpg = require("./models/mpg");

mpg.run();

const app = express();

app.use(express.static(__dirname + "/public"));

app.get("/", (request, response) => {
  response.sendFile(__dirname + "/views/index.html");
});

app.get("/model", (request, response) => {
  response.json({
    mean: mpg.mean.arraySync(),
    variance: mpg.variance.arraySync(),
    mpg: {
      timestamp: mpg.timestamp,
      weights: mpg.weights.arraySync(),
      r2: mpg.r2
    }
  });
});

const listener = app.listen(process.env.PORT, () => {
  console.log("Your app is listening on port " + listener.address().port);
});
