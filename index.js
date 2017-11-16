var math = require('mathjs');
var stat = require('pw-stat');

/**
 * An LDA object.
 * @constructor
 * @param {number[]} set1 - Data set for class 1, rows are samples, columns are variables
 * @param {number[]} set2 - Data set for class 2, rows are samples, columns are variables
 */
function LDA(set1, set2) {
	var mu1 = math.transpose(stat.mean(set1));
	var mu2 = math.transpose(stat.mean(set2));
	var pooledCov = math.add(stat.cov(set1), stat.cov(set2));
	theta = math.multiply(math.inv(pooledCov), math.subtract(mu2, mu1));
	b = math.multiply(-1, math.transpose(theta), math.add(mu1, mu2), 1 / 2);

	this.theta = theta;
	this.b = b;
}

/**
 * Predict the class of an unknown data point.
 * @param {number[]} vector - The data point to be classified. Should be a one-dimensional array.
 * @returns {number} - less than 0 if class 1, 0 if exactly inbetween, greater than 0 if class 2
 */
LDA.prototype.project = function (vector) {
	return math.add(math.multiply(vector, this.theta), this.b);
}

module.exports = LDA;
