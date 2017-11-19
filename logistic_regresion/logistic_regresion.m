1;

function g = sigmoid(z)
  g = 1 ./ (1 + e.^-z);
end

function [J, grad] = costFunction(theta, X, y)

  J = (1/length(y))*(-y'* log(sigmoid(X*theta)) - (1 - y)'* log(1-sigmoid(X*theta)));
  grad = (1/length(y))*X'*(sigmoid(X*theta) - y);

end

function p = predict(theta, X)

  m = size(X, 1);
  p = zeros(m, 1);
  p = sigmoid(X*theta) >= 0.5;

end



clear ; close all; clc


data =csvread('winequality-red_cleaned.csv');
X = data(:, [1, 11]); y = data(:, 12);

[m, n] = size(X);
X = [ones(m, 1) X];
initial_theta = zeros(n + 1, 1);

[cost, grad] = costFunction(initial_theta, X, y);

fprintf('Cost at initial theta (zeros): %f\n', cost);
fprintf('Gradient at initial theta (zeros): \n');
fprintf(' %f \n', grad);

options = optimset('GradObj', 'on', 'MaxIter', 400);

[theta, cost] = ...
	fminunc(@(t)(costFunction(t, X, y)), initial_theta, options);

fprintf('Cost at theta found by fminunc: %f\n', cost);
fprintf('theta: \n');
fprintf(' %f \n', theta);

p = predict(theta, X);
fprintf('Train Accuracy: %f\n', mean(double(p == y)) * 100);



