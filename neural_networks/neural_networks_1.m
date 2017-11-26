1;

function g = sigmoid(z)
  g = 1.0 ./ (1.0 + exp(-z));
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
  a1 = [ones(m,1) X];
  z2 = a1 *Theta1';
  a2 = [ones(size(z2),1) sigmoid(z2)];
  z3 = a2*Theta2';
  a3 = sigmoid(z3);
  [predict_max, index_max] = max(a3, [], 2);

  p = index_max;

end

clear ; close all; clc

input_layer_size  = 400; 
hidden_layer_size = 25;   

load('data1.mat');
m = size(X, 1);
sel = randperm(size(X, 1));
sel = sel(1:100);

displayData(X(sel, :));


load('weights.mat');

pred = predict(Theta1, Theta2, X);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);
rp = randperm(m);

for i = 1:m
   
    fprintf('\nDisplaying Example Image\n');
    displayData(X(rp(i), :));

    pred = predict(Theta1, Theta2, X(rp(i),:));
    fprintf('\nNeural Network Prediction: %d (digit %d)\n', pred, mod(pred, 10));
    s = input('Paused - press enter to continue, q to exit:','s');
    if s == 'q'
      break
    end
end
