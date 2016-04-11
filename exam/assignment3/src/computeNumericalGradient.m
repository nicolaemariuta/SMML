function numgrad = computeNumericalGradient( X,y,w1,w2,hiddenLayerSize, ...
    inputLayerSize,inputBiasSize, outputBiasSize, outputLayerSize)
rw1 = reshape (w1,[1,size(w1,1)*size(w1,2)]);
rw2 = reshape (w2,[1,size(w2,1)*size(w2,2)]);
paramsInitial = [rw1,rw2];

perturb = zeros(size(paramsInitial));
numgrad = zeros(size(paramsInitial));
e = 10^(-4);

for p = 1:length(paramsInitial)
    %set perturbation vector
    perturb(p) = e;
       
    [w_1, w_2] = reshapeFunction(paramsInitial + perturb, hiddenLayerSize, ...
        inputLayerSize,inputBiasSize, outputBiasSize, outputLayerSize);
    
    loss2 = costFunction(X, y, w_1, w_2);
    
    [w_1, w_2] = reshapeFunction(paramsInitial - perturb, hiddenLayerSize, ...
        inputLayerSize,inputBiasSize, outputBiasSize, outputLayerSize);
    
    loss1 = costFunction(X, y, w_1, w_2);
    
    %compute numerical gradient
    numgrad(p) = (loss2 - loss1)/(2*e);
    
    perturb(p) = 0;
    
end

end

