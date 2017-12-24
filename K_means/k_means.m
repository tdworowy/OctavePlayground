1;
function centroids = InitCentroids(X, K)
 randidx = randperm(size(X, 1));
 centroids = X(randidx(1:K), :);
end

function idx = findClosestCentroids(X, centroids)
  K = size(centroids, 1);
  idx = zeros(size(X,1), 1);
  for i = 1:size(X, 1)
    min = inf;
    for j=1:K
      distance = (X(i, :)'-centroids(j, :)');
      distance = distance'*distance;
      if distance < min
        min = distance;
        idx(i) = j;
       end  
  end
    end
end

function [X_norm, mu, sigma] = featureNormalize(X)

  mu = mean(X);
  X_norm = bsxfun(@minus, X, mu);

  sigma = std(X_norm);
  X_norm = bsxfun(@rdivide, X_norm, sigma);

end

function centroids = computeCentroids(X, idx, K)
  [m n] = size(X);
  for k = 1:K
    num_k = 0;
    sum = zeros(n, 1);
    for i = 1:m
      if ( idx(i) == k )
        sum = sum + X(i, :)';
        num_k = num_k + 1;
      end
    end
    centroids(k, :) = (sum/num_k)';
  end
end


function [centroids, idx] = runkMeans(X, initial_centroids,  max_iters, plot_progress)

  [m n] = size(X);
  K = size(initial_centroids, 1);
  centroids = initial_centroids;
  previous_centroids = centroids;
  idx = zeros(m, 1);
  for i=1:max_iters

      idx = findClosestCentroids(X, centroids);
      centroids = computeCentroids(X, idx, K);
  end

end


clear ; close all; clc
fprintf('loading data...\n')
fflush(stdout)
A = double(imread('cat.jpg'));
fprintf('DONE\n')
fflush(stdout)

A = A / 255; 

img_size = size(A);
img_size
fflush(stdout)

X = reshape(A, img_size(1) * img_size(2), 3);
K = 16; 
max_iters = 10;

initial_centroids = InitCentroids(X, K);

[centroids, idx] = runkMeans(X, initial_centroids, max_iters);

idx = findClosestCentroids(X, centroids);

X_recovered = centroids(idx,:);

X_recovered = reshape(X_recovered, img_size(1), img_size(2), 3);

subplot(1, 2, 1);
imagesc(A); 
title('Original');

subplot(1, 2, 2);
imagesc(X_recovered)
title(sprintf('Compressed, with %d colors.', K));
