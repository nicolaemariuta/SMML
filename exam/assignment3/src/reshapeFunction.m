function [ w1,w2 ] = reshapeFunction( vector, hiddenLayerSize, ... 
    inputLayerSize,inputBiasSize, outputBiasSize, outputLayerSize)
%RESHAPEFUNCTION Summary of this function goes here
%   Detailed explanation goes here
w1_start = 1;
w1_end = hiddenLayerSize * (inputLayerSize + inputBiasSize);
w1 = reshape(vector(w1_start:w1_end), [inputLayerSize + inputBiasSize ,hiddenLayerSize]);

w2_end = w1_end + (hiddenLayerSize+ outputBiasSize) * outputLayerSize ;
w2 = reshape(vector(w1_end+1:w2_end),[hiddenLayerSize + outputBiasSize, outputLayerSize]);      
end

