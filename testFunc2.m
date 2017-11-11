#can't run it directly
#can use only function with same name as file
function [a,b,c] = testFunc1(x)
  a = x ^2
  b = x ^3
  c = x ^4
  
function [a,b,c] = testFunc2(x)
  [a,b,c ]= 2 * testFunc1(x)

 
