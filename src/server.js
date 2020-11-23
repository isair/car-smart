const express = require('express');

const serializeModelResult = require('./utils/serializeModelResult');
const mpgModel = require('./models/mpg');
const emissionsModel = require('./models/passedemissions');
const environment = require('./environment');

const start = () => {
  // Run models and serialize results.

  const serializeModel = (model) => serializeModelResult(model.run());

  const serializedModelResults = {
    mpg: serializeModel(mpgModel),
    emissions: serializeModel(emissionsModel),
  };

  // Set up and start the web server.

  const app = express();

  app.use(express.static(`${__dirname}/public`));

  app.get('/', (_request, response) => {
    response.sendFile(`${__dirname}/views/index.html`);
  });

  app.get('/model', (request, response) => response.json(serializedModelResults[request.query.name || 'mpg']));

  const listener = app.listen(environment.port, () => {
    // eslint-disable-next-line no-console
    console.log(
      environment.dev
        ? `You can access your app from http://localhost:${listener.address().port}`
        : `Your app is listening on port ${listener.address().port}`,
    );
  });
};

module.exports = {
  start,
};
