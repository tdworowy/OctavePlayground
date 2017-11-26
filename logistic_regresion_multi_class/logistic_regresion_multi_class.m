1;

function g = sigmoid(z)
  g = 1.0 ./ (1.0 + exp(-z));
end


function [J, grad] = lrCostFunction(theta, X, y, lambda)
  m = length(y); 
  J = 0;
  grad = zeros(size(theta));
  h = sigmoid(X*theta);
  theta_reg = [0;theta(2:end, :);];
  J = (1/m)*(-y'* log(h) - (1 - y)'*log(1-h))+(lambda/(2*m))*theta_reg'*theta_reg;
  grad = (1/m)*(X'*(h-y)+lambda*theta_reg);

end


function [all_theta] = oneVsAll(X, y, num_labels, lambda)
  m = size(X, 1);
  n = size(X, 2);
  all_theta = zeros(num_labels, n + 1);
  X = [ones(m, 1) X];
  initial_theta = zeros(n + 1, 1);
  options = optimset('GradObj', 'on', 'MaxIter', 50);
  for c = 1:num_labels,
        [theta] = fminunc(@(t)(lrCostFunction(t, X, (y == c), lambda)),initial_theta, options);
        all_theta(c, :) = theta';
  end
end


function p = predictOneVsAll(all_theta, X)
  m = size(X, 1);
  num_labels = size(all_theta, 1);
  p = zeros(size(X, 1), 1);
  X = [ones(m, 1) X];
  predict = sigmoid(X*all_theta');
  [predict_max, index_max] = max(predict, [], 2);
  p = index_max;

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


clear ; close all; clc

input_layer_size  = 400;  
num_labels = 10;          

load('data1.mat'); 
m = size(X, 1);

rand_indices = randperm(m);
sel = X(rand_indices(1:100), :);

displayData(sel);

theta_t = [-2; -1; 1; 2];
X_t = [ones(5,1) reshape(1:15,5,3)/10];
y_t = ([1;0;1;0;1] >= 0.5);
lambda_t = 3;
[J grad] = lrCostFunction(theta_t, X_t, y_t, lambda_t);

fprintf('\nCost: %f\n', J);
fprintf('Expected cost: 2.534819\n');
fprintf('Gradients:\n');
fprintf(' %f \n', grad);
fprintf('Expected gradients:\n');
fprintf(' 0.146561\n -0.548558\n 0.724722\n 1.398003\n');

fprintf('\nTraining One-vs-All Logistic Regression...\n')

lambda = 0.1;
[all_theta] = oneVsAll(X, y, num_labels, lambda);

pred = predictOneVsAll(all_theta, X);
fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);
