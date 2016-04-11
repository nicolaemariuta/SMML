function numgrad = computeNumericalGradient( X,y,w1,w2 )
rw1 = reshape (w1,[1,size(w1,1)*size(w1,2)]);
rw2 = reshape (w2,[1,size(w2,1)*size(w2,2)]);
paramsInitial = [rw1,rw2];
numgrad = zeros(size(paramsInitial));
e = 10^(-4);

for p = 1:length(paramsInitial)
    %set perturbation vector
    w_1 = w1;
    w_2 = w2;
    if(p<(length(w1)+1))
        w_1(1,p) = w_1(1,p) + e;
    else
        w_2(p-3,1) = w_2(p-3,1) + e;
    end
    loss2 = costFunction(X, y, w_1, w_2);
    
    w_11 = w1;
    w_12 = w2;
    if(p<(length(w1)+1))
        w_11(1,p) = w_11(1,p) - e;
    else
        w_12(p-3,1) = w_12(p-3,1) - e;
    end
    loss1 = costFunction(X, y, w_11, w_12);
    
    %compute numerical gradient
    numgrad(p) = (loss2 - loss1)/(2*e);
    
end

end

