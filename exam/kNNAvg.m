function rMS  = kNNAvg( K , TrainMat, TestMat )
%function for calculating the error for the behaviour of the knn model used
%together with the test set. Calculates averaged sum of k-nearest
%neighbours for each pattern in TestMat by using the patterns from TrainMat

%number of paramteters in pattern(without the last column)
nrParams = size(TestMat,2)-1;

%matrix that will hold the predictions for each pattern in test set
predictions = zeros(size(TestMat,1),1);
%calcualte regression taking average for all k nearest neighbours
for i = 1:size(TestMat,1)
    %calcualte distance from test pattern to each patter in train set
    distances = zeros(size(TrainMat,1),2);
    for j = 1:size(TrainMat,1)
        d = sqrt(sum(power(TrainMat(j,1:nrParams)-TestMat(i,1:nrParams),2)));
        distances(j,:) = [d,TrainMat(j,nrParams+1)]; 
    end
    %sort distances to take the first k
    distances = sortrows(distances,1);
    
    %calcualte sum for redshift values of nearest k neighbours in train set
    sumNearest = 0;
    for k = 1:K 
        sumNearest = sumNearest + distances(k,2);
    end
    %calculate avergae or redshift and insert into predictions matrix
    predictions(i) = sumNearest/size(TrainMat,1);
    
    
end

%calcualte RMS
rMS = my_ms(predictions,TestMat(:,nrParams+1));

end

