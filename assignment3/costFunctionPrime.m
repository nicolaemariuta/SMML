function [dJW1,dJW2] = costFunctionPrime( X,y,w1,w2 )

[z2,a2,z3,yHat] = forward(X,w1,w2);


delta3 = (-(y-yHat)).*activationFunctionDerivate(z3);
dJW2=transpose(a2)*delta3;

delta2 = (delta3*transpose(w2)).*activationFunctionDerivate(z2);
dJW1 = transpose(X)*delta2;
end

