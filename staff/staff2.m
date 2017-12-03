v = rand(7,1);
w = rand(7,1);

z = 0;
for i = 1:7
  z = z + v(i) * w(i);
end
z
z1 = sum (v .* w);
z1
z2 = v' * w;
z2
z3 =  v * w';
z3
z4 = z = v .* w;
z4