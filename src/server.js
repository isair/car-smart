const express = require("express");
const mpgModel = require("./models/mpg");

const mpg = mpgModel.run();

const app = express();

app.use(express.static(__dirname + "/public"));

app.get("/", (request, response) => {
  response.sendFile(__dirname + "/views/index.html");
});

app.get("/model", (request, response) => {
  switch (request.query.name) {
    case "mpg":
    default:
      response.json({
        mean: mpg.mean.arraySync(),
        variance: mpg.variance.arraySync(),
        weights: mpg.weights.arraySync(),
        r2: mpg.r2,
        msePlotImageUrl: mpg.msePlotImageUrl
      });
      break;
  }
});

const listener = app.listen(process.env.PORT, () => {
  console.log("Your app is listening on port " + listener.address().port);
});
