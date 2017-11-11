X = rand(7,7);
for i = 1:7
  for j = 1:7
    A(i, j) = log(X(i, j));
    B(i, j) = X(i, j) ^ 2;
    C(i, j) = X(i, j) + 1;
    D(i, j) = X(i, j) / 4;
  end
end

A
B
C
D

A1 = log (X);
B1 = X ^ 2;
C1 =  X + 1;
D1 = X / 4;

A1
B1
C1
D