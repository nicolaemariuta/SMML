function [ riskSum ] = kNN( K , TrainMat, TestMat )
%KNN Summary of this function goes here
%   Detailed explanation goes here

riskSum = 0;

for i = 1:size(TestMat,1)
    distances = zeros(size(TrainMat,1),2);
    for j = 1:size(TrainMat,1)
        d = sqrt(power(TrainMat(j,1)-TestMat(i,1),2)+power(TrainMat(j,2)-TestMat(i,2),2));
        distances(j,:) = [d,TrainMat(j,3)]; 
    end
    distances = sortrows(distances,1);

    countingVector = zeros(1,3);
    for k = 1:K 
        countingVector(distances(k,2) + 1) = countingVector(distances(k,2) + 1) + 1;
    end
    
    index = find(countingVector == max(countingVector(:))) - 1;
  
    if index ~= TestMat(i,3)
        riskSum = riskSum + 1;
    end
    
end

riskSum = riskSum / size(TestMat,1);

end

