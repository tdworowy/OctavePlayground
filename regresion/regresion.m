1;


function [X_norm, mu, sigma] = featureNormalize(X)

  X_norm = X;
  mu = zeros(1, size(X, 2));
  sigma = zeros(1, size(X, 2));
      
  mu = mean(X,1)
  sigma = std(X,0,1)
  X_norm = (X .- mu) ./ sigma;


end


function J = computeCost(X, y, theta)

  m = length(y);
  J = 0;
  H = (theta'*X')';
  S = sum((H - y) .^ 2);
  J = S / (2*m);
end



function [theta, history] = gradientDescent(X, y, theta, alpha, num_iters)

  m = length(y);
  history = zeros(num_iters, 1);

  for iter = 1:num_iters,
     theta = theta -((1/m) * ((X * theta) - y)' * X)' * alpha;
     history(iter) = computeCost(X, y, theta);

  end
end


clear ; close all; clc

data = csvread('carprice.txt');
X = data(:,[3:3 5:10]);
y = data(:, 4);
m = length(y);


[X mu sigma] = featureNormalize(X);
X = [ones(m, 1) X];


alpha = 0.01;
num_iters = 400;
 
theta = zeros(8, 1);
[theta, history] = gradientDescent(X, y, theta, alpha, num_iters);

figure;
plot(1:numel(history), history, '-b', 'LineWidth', 2);
xlabel('Iterations');
ylabel('Cost J');

fprintf('Theta: \n');
fprintf(' %f \n', theta);
fprintf('\n');


fprintf('TEST\n');
predict = [40;80;40;40;30;30;40];
[predict mu sigma] = featureNormalize(predict);
predict = [1;predict];
resoult = 0;
for i=1:8,
  resoult = resoult + (predict(i) * theta(i));
end;
resoult





