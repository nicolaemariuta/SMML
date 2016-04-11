function [ bestC,bestGamma ] = my_fiveFoldGaussianKernelCV( trainX,trainY,Cs,gammas, blockDimensions)
%MY_FIVEFOLDCV apply cross validation by building SVM gaussian kernel for the train set using the set of Cs
%and gammas that are given. BlockDimensions is the dimension for the blocks
%that will be used for splittinf the data

%store the average accuracy for each model that is obtained
accs = zeros(length(Cs),length(gammas));


%create models for all combination between gammas and Cs
for i = 1:length(Cs)
    currentC = Cs(i);
    
    for j = 1:length(gammas);
        %create kernel function for current value of gamma
        kernel = @(x,y) my_kernel(x, y, gammas(j));
        acc = zeros(1,5);
        
        %take each of the 5 sections out and use it as test block while the
        %others are used to train the model
        for s = 1:5
            %split into blocks
            testXBlock = trainX((s-1)*blockDimensions(s) + 1:s*blockDimensions(s),:);
            testYBlock = trainY((s-1)*blockDimensions(s) + 1:s*blockDimensions(s),:);
            
            TrainX = trainX;
            TrainX((s-1)*blockDimensions(s) + 1:s*blockDimensions(s),:) = [];
            TrainY = trainY;
            TrainY((s-1)*blockDimensions(s) + 1:s*blockDimensions(s),:) = [];            
           
            %train model, calcualte predictions of model and store the result for
            %accuracy
            model = svmtrain(TrainX, TrainY, ...
                'autoscale',false, ... 
                'boxconstraint',ones(size(TrainX,1),1) * currentC, ...
                'kernel_function',kernel);
    
            predY = svmclassify(model,testXBlock);
            acc(s) = 1 - (nnz(predY - testYBlock)) / length(predY);
        end
        %store the average accucarcy for all the s blocks
        accs(i,j) = mean(acc);
    end
end
%find the value for Gamma and C that has given the best accuracy
disp(accs);
[row,col] = find(accs == max(accs(:)));
bestC = Cs(row(1));
bestGamma = gammas(col(1));

end



