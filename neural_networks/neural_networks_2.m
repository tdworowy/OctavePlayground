1;

function g = sigmoid(z)
  g = 1.0 ./ (1.0 + exp(-z));
end


function g = sigmoidGradient(z)
 g = sigmoid(z) .* (1- sigmoid(z));
end


function W = randInitializeWeights(L_in, L_out)
  W = zeros(L_out, 1 + L_in);
  epsilon_init = 0.12;
  W = rand(L_out, 1 + L_in) * 2 * epsilon_init - epsilon_init;
end



function [h, display_array] = displayData(X, example_width)

  if ~exist('example_width', 'var') || isempty(example_width) 
    example_width = round(sqrt(size(X, 2)));
  end

  colormap(gray);

  [m n] = size(X);
  example_height = (n / example_width);

  display_rows = floor(sqrt(m));
  display_cols = ceil(m / display_rows);
  pad = 1;
  display_array = - ones(pad + display_rows * (example_height + pad), ...
                         pad + display_cols * (example_width + pad));

  curr_ex = 1;
  for j = 1:display_rows
    for i = 1:display_cols
      if curr_ex > m, 
        break; 
      end
      max_val = max(abs(X(curr_ex, :)));
      display_array(pad + (j - 1) * (example_height + pad) + (1:example_height), ...
                    pad + (i - 1) * (example_width + pad) + (1:example_width)) = ...
              reshape(X(curr_ex, :), example_height, example_width) / max_val;
      curr_ex = curr_ex + 1;
    end
    if curr_ex > m, 
      break; 
    end
  end

  h = imagesc(display_array, [-1 1]);
  axis image off

  drawnow;

end


function p = predict(Theta1, Theta2, X)
  m = size(X, 1);
  num_labels = size(Theta2, 1);

  p = zeros(size(X, 1), 1);

  h1 = sigmoid([ones(m, 1) X] * Theta1');
  h2 = sigmoid([ones(m, 1) h1] * Theta2');
  [dummy, p] = max(h2, [], 2);

end

function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
  Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                   hidden_layer_size, (input_layer_size + 1));

  Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                   num_labels, (hidden_layer_size + 1));

  m = size(X, 1);

  J = 0;
  Theta1_grad = zeros(size(Theta1));
  Theta2_grad = zeros(size(Theta2));

  I = eye(num_labels);
  Y = zeros(m, num_labels);
  for i=1:m
    Y(i, :)= I(y(i), :);
  end


  A1 = [ones(m, 1) X];
  Z2 = A1 * Theta1';
  A2 = [ones(size(Z2, 1), 1) sigmoid(Z2)];
  Z3 = A2*Theta2';
  H = A3 = sigmoid(Z3);


  theta1_sum = sum(sum(Theta1(:, 2:end).^2, 2));
  theta2_sum = sum(sum(Theta2(:,2:end).^2, 2));

  reqularization = (lambda/(2*m))*(theta1_sum + theta2_sum);

  J = (1/m)*sum(sum((-Y).*log(H) - (1-Y).*log(1-H), 2))+ reqularization;


  Sigma3 = A3 - Y;
  Sigma2 = (Sigma3*Theta2 .* sigmoidGradient([ones(size(Z2, 1), 1) Z2]))(:, 2:end);


  Delta_1 = Sigma2'*A1;
  Delta_2 = Sigma3'*A2;


  Theta1_grad = Delta_1./m + (lambda/m)*[zeros(size(Theta1,1), 1) Theta1(:, 2:end)];
  Theta2_grad = Delta_2./m + (lambda/m)*[zeros(size(Theta2,1), 1) Theta2(:, 2:end)];
  grad = [Theta1_grad(:) ; Theta2_grad(:)];


end

function numgrad = computeNumericalGradient(J, theta)
  numgrad = zeros(size(theta));
  perturb = zeros(size(theta));
  e = 1e-4;
  for p = 1:numel(theta)
      perturb(p) = e;
      loss1 = J(theta - perturb);
      loss2 = J(theta + perturb);
      numgrad(p) = (loss2 - loss1) / (2*e);
      perturb(p) = 0;
  end

end

function W = debugInitializeWeights(fan_out, fan_in)
  fprintf('debugInitializeWeights.\n');
  W = zeros(fan_out, 1 + fan_in);
  W = reshape(sin(1:numel(W)), size(W)) / 10;

end



function checkNNGradients(lambda)

  if ~exist('lambda', 'var') || isempty(lambda)
      lambda = 0;
  end

  input_layer_size = 3;
  hidden_layer_size = 5;
  num_labels = 3;
  m = 5;

  Theta1 = debugInitializeWeights(hidden_layer_size, input_layer_size);
  Theta2 = debugInitializeWeights(num_labels, hidden_layer_size);

  X  = debugInitializeWeights(m, input_layer_size - 1);
  y  = 1 + mod(1:m, num_labels)';

  nn_params = [Theta1(:) ; Theta2(:)];

  costFunc = @(p) nnCostFunction(p, input_layer_size, hidden_layer_size, ...
                                 num_labels, X, y, lambda);

  [cost, grad] = costFunc(nn_params);
  numgrad = computeNumericalGradient(costFunc, nn_params);

  disp([numgrad grad]);

end

clear; close all; clc

input_layer_size  = 400;  
hidden_layer_size = 25;   
num_labels = 10;          

fprintf('Load data1.mat.\n');
load('data1.mat');
m = size(X, 1);

sel = randperm(size(X, 1));
sel = sel(1:100);

displayData(X(sel, :));

fprintf('Load weights.mat.\n');
load('weights.mat');

nn_params = [Theta1(:) ; Theta2(:)];


lambda = 1;
fprintf('nnCostFunction.\n');
J = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, ...
                   num_labels, X, y, lambda);

g = sigmoidGradient([-1 -0.5 0 0.5 1]);


initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);

initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];

fprintf('checkNNGradients.\n');
checkNNGradients;

lambda = 3;
checkNNGradients(lambda);

fprintf('nnCostFunction.\n');
debug_J  = nnCostFunction(nn_params, input_layer_size, ...
                          hidden_layer_size, num_labels, X, y, lambda);

options = optimset('MaxIter', 50);

lambda = 1;
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X, y, lambda);
fprintf('fminunc.\n');
[nn_params, cost] = fminunc(costFunction, initial_nn_params, options);

Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

displayData(Theta1(:, 2:end));

pred = predict(Theta1, Theta2, X);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);
