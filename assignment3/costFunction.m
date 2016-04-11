function J = costFunction( X, y, w1, w2 )
%Compute cost for given X,y, use weights already stored in class.
[z2,a2,z3,yHat] = forward(X,w1,w2);

J = 0.5*sum(((y-yHat).^2));

end

