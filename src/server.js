const express = require("express");

const serializeModelResult = require("./utils/serializeModelResult");
const mpgModel = require("./models/mpg");

// Run models and serialize results.

const serializedModelResults = {
  mpg: serializeModelResult(mpgModel.run())
};

// Set up and start the web server.

const app = express();

app.use(express.static(__dirname + "/public"));

app.get("/", (request, response) => {
  response.sendFile(__dirname + "/views/index.html");
});

app.get("/model", (request, response) =>
  response.json(serializedModelResults[request.query.name || "mpg"])
);

const listener = app.listen(process.env.PORT, () => {
  console.log("Your app is listening on port " + listener.address().port);
});
