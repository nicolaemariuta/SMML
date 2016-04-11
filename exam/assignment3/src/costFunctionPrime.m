function [dJW1,dJW2] = costFunctionPrime( X,y,w1,w2 )

[z2,a2,z3,yHat] = forward(X,w1,w2);

bias = ones([size(X,1),1]);
input = horzcat(X,bias);

div = 2/size(X,1);

delta3 = div*(-(y-yHat)).*activationFunctionDerivate(z3);
dJW2=transpose(a2)*delta3;

delta2 = div*(delta3*transpose(w2)).*activationFunctionDerivate(z2);
delta2 = delta2(:,1:2);

dJW1 = transpose(input)*delta2;
end

