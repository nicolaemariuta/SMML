function [z2,a2,z3,yHat] = forward( X, w1, w2 )
%forward propagation
bias = zeros(size(X,1)) +1;
input = horzcat(X,bias);
disp(input);
z2 = input*w1;
a2 = activationFunction(z2);
a2 = horzcat(a2,bias);
z3 = a2*w2;
yHat = activationFunction(z3);
end

