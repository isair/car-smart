const developmentEnvironment = require('./environment.development');
const productionEnvironment = require('./environment.production');

module.exports = process.env.NODE_ENV === 'development' ? developmentEnvironment : productionEnvironment;
