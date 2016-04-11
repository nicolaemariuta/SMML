function [ bestC,bestGamma ] = my_fiveFoldCV( trainX,trainY,Cs,gammas)
%MY_FIVEFOLDCV Summary of this function goes here
%   Detailed explanation goes here

accs = zeros(length(Cs),length(gammas));

blockDimensions = [20,20,20,20,17];

for i = 1:length(Cs)
    currentC = Cs(i);
    for j = 1:length(gammas);
        sigma = sqrt(1/(2*gammas(j)));
        acc = zeros(1,5);
        for s = 1:5
            trainXBlock = trainX((s-1)*blockDimensions(s) + 1:s*blockDimensions(s),:);
            trainYBlock = trainY((s-1)*blockDimensions(s) + 1:s*blockDimensions(s),:);
            
            TestX = trainX;
            TestX((s-1)*blockDimensions(s) + 1:s*blockDimensions(s),:) = [];
            TestY = trainY;
            TestY((s-1)*blockDimensions(s) + 1:s*blockDimensions(s),:) = [];            
            
            model = svmtrain(trainXBlock, trainYBlock, 'kernel_function','rbf','rbf_sigma',sigma,'boxconstraint',currentC,'autoscale',false);
            predY = svmclassify(model,TestX);
            acc(s) = 1 - (nnz(predY - TestY)) / length(predY);
        end
       
        accs(i,j) = mean(acc);
    end
end

[row,col] = find(accs == max(accs(:)));
bestC = Cs(row(1));
bestGamma = gammas(col(1));

end

