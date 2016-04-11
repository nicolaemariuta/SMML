function bestC = my_fiveFoldCV( trainX,trainY,Cs, blockDimensions)
%MY_FIVEFOLDCV apply cross validation for the train set using the set of Cs
%and gammas that are given. BlockDimensions is the dimension for the blocks
%that will be used for splittinf the data

%store the average accuracy for each model that is obtained
accs = zeros(length(Cs),1);


%create models for all combination between gammas and Cs
for i = 1:length(Cs)
    currentC = Cs(i);
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
            model = svmtrain(TrainX, TrainY, 'kernel_function','linear','boxconstraint',currentC,'autoscale',false);
            predY = svmclassify(model,testXBlock);
            acc(s) = 1 - (nnz(predY - testYBlock)) / length(predY);
        end
        %store the average accucarcy for all the s blocks
        accs(i,1) = mean(acc);
    
end
%find the value for Gamma and C that has given the best accuracy
disp(accs);
row = find(accs == max(accs(:)));
bestC = Cs(row(1));


end

