function [ endErrors, validErrors ] = train( xTrain, yTrain, xTest, yTest, w1, w2, ...
    epoch, minErr, hiddenLayerSize, inputLayerSize,inputBiasSize, ...
    outputBiasSize, outputLayerSize)
%TRAIN Summary of this function goes here
%   Detailed explanation goes here

diff = Inf;
iter = 0;

endErrors = [];
validErrors = [];

while diff > minErr && iter < epoch
    iter = iter + 1;
    backPropGradients = [];
    
    beginError = costFunction(xTrain,yTrain,w1,w2);
    
    for i=1:size(xTrain,1)
        x = xTrain(i,:);
        y = yTrain(i,:);
        bpGradient = computeGradients(x,y,w1,w2);
        backPropGradients = [backPropGradients, bpGradient];
    end
    
    sumGrad = sum(backPropGradients);
    scalar = 5e-4;   
    
    w_1 = reshape (w1,[1,size(w1,1)*size(w1,2)]);
    w_2 = reshape (w2,[1,size(w2,1)*size(w2,2)]);
    currGradient = [w_1, w_2];
    
    newGradient = currGradient - scalar*sumGrad;
    
    [w1,w2] = reshapeFunction(newGradient,hiddenLayerSize, ...
    inputLayerSize,inputBiasSize, outputBiasSize, outputLayerSize);
    
    endError = costFunction(xTrain,yTrain,w1,w2);
    validError = mean(costFunction(xTest,yTest,w1,w2));
    endErrors = [endErrors,endError];    
    validErrors = [validErrors, validError];
    
    diff = abs(endError - beginError);
       
    
end
end

