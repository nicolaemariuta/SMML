function prediction = my_lmfunction( wML, X )
%MY_ Summary of this function goes here
%   Detailed explanation goes here 
    M = X';
    M = vertcat(ones(1,size(X,1)), M);
    prediction = wML'*M;
    prediction = prediction';

end

