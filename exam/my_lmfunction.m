function prediction = my_lmfunction( wML, X )
%Calculate prediction for the test data X based on the linear model wML
    M = X';
    M = vertcat(ones(1,size(X,1)), M);
    prediction = wML'*M;
    prediction = prediction';

end

