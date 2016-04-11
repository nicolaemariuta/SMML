function [ riskSum ] = kNN( K , TrainMat, TestMat )
%KNN nearest neighbour, calcualte risksum for input data and k
nrParams = size(TestMat,2)-1;

riskSum = 0;

for i = 1:size(TestMat,1)
    %calculate distance from each pattern in test set towards each poin in
    %train set
    distances = zeros(size(TrainMat,1),2);
    for j = 1:size(TrainMat,1)
        d = sqrt(sum(power(TrainMat(j,1:nrParams)-TestMat(i,1:nrParams),2)));
        distances(j,:) = [d,TrainMat(j,nrParams+1)]; 
    end
    distances = sortrows(distances,1);
    %find the patterns that appears in largest number in the neighbourhood
    countingVector = zeros(1,5);
    for k = 1:K 
        countingVector(distances(k,2)) = countingVector(distances(k,2)) + 1;
    end
    
    index = find(countingVector == max(countingVector(:)));
    %calculate risk sum
    if index ~= TestMat(i,nrParams+1)
        riskSum = riskSum + 1;
    end
    
end

riskSum = riskSum / size(TestMat,1);

end

