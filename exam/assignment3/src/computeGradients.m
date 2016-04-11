function Gr = computeGradients( X,y,w1,w2 )
[dJW1,dJW2] = costFunctionPrime(X,y,w1,w2);
rdJW1 = reshape (dJW1,[1,size(dJW1,1)*size(dJW1,2)]);
rdJW2 = reshape (dJW2,[1,size(dJW2,1)*size(dJW2,2)]);
Gr = [rdJW1, rdJW2];

end

