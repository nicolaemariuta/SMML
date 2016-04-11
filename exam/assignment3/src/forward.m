function [z2,a2,z3,yHat] = forward( X, w1, w2)
%forward propagation

bias = ones([size(X,1),1]);
input = horzcat(X,bias);

z2 = input*w1;  

bias2 = ones([size(z2,1),1]);
z2 = horzcat(z2,bias2);

a2 = activationFunction(z2);
z3 = a2*w2;
yHat = activationFunction(z3);
end

