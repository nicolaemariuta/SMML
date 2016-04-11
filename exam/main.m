%Question 1 (linear regression)
clear;fprintf('Question 1 - linear regression \n');

%read data from files with train and test variables
trainData = importdata('data/redshiftTrain.csv');
testData = importdata('data/redshiftTest.csv');
trainData = trainData(1:500,:);
l = length(trainData);

%Build linear model
Phi1 = horzcat(ones(l,1),trainData(:,1:10));
wML = my_wML(Phi1, trainData(:,11));
disp('linear model:')
disp(wML);

%Calculate prediction for train data
X1 = trainData (:,1:10);
predictionTrain = my_lmfunction(wML,X1);

%Calculate prediction for test data
X1 = testData (:,1:10);
predictionTest = my_lmfunction(wML,X1);

%Calculate mean-squared error for train set
ms = my_ms(predictionTrain,trainData(:,11));
disp('mean-squared error for train set')
disp(ms);

%Calculate mean-squared error for test set
ms = my_ms(predictionTest,testData(:,11));
disp('mean-squared error for test set')
disp(ms);
%plot redshift in test data vs prediction
figure;
hold on;
title('Question 1 - Calculated redshift(blue) vs. predicted(red) for test data','FontSize', 15);
plot(100:200, predictionTest(100:200,1), 'r', 100:200, testData(100:200,11), 'b');
legend('Predicted redshift', 'Calculated redshift');
hold off;

%%
%Question 2 (non-linear regression) 
%Method 2: KNN
clearvars();
fprintf('Question 2 - non-linear regression by using k-nearest neighbor\n');
%read and normalize data
trainData = importdata('data/redshiftTrain.csv');
testData = importdata('data/redshiftTest.csv');

Means = mean(trainData(:,1:10), 1);
Stds = std(trainData(:,1:10), 1, 1);

normalizedTrainData = my_normalize(trainData, Means, Stds,1,10,11);
normalizedTestData = my_normalize(testData, Means, Stds,1,10,11);

%number of sections for cross-validation and block dimension
S = 5;
blockDimension = int32(size(normalizedTrainData,1)/S)

%matrix that will contain k and corresponding average RMS
kMSMatrix = [];

%make cross validation for several values of k
for k = 1:1:15
    %sum of RMS for each section
    kAverageMS = 0;
    
   for s = 1:S
        %split data 
        Test = normalizedTrainData((s-1)*blockDimension + 1:s*blockDimension,:);
        Train = normalizedTrainData;
        Train((s-1)*blockDimension + 1:s*blockDimension,:) = [];
        %calcualte RMS for using knn with value k
        kAverageMS = kAverageMS + kNNAvg(k,Train, Test); 
   end
   %add pairs with value of k and average RMS 
   new_row =[k kAverageMS/5] ; 
   kMSMatrix = [kMSMatrix ; new_row];

end

%take the k with the min error
kBestIndex = find(kMSMatrix(:,2) == min(kMSMatrix(:,2)));
kBest = kMSMatrix(kBestIndex,1);


%calcualte RMS for train dataset by applying knn with best k that was found through crossvalidation    
eRMStrain = kNNAvg(kBest,normalizedTrainData,normalizedTrainData);
disp('mean-squared-error for the train set: ')
disp(eRMStrain);
    
%calcualte RMS for test dataset by applying knn with best k that was found through crossvalidation    
eRMStest = kNNAvg(kBest,normalizedTrainData,normalizedTestData);
disp('mean-squared-error for the test set: ')
disp(eRMStest);   


%%
%Question 2 (non-linear regression) 
%Method 3: Random forest
clearvars();
fprintf('Question 2 - non-linear regression by using Random Forest\n');

%read and split into vector for paramters and vector for redshift 
trainData = importdata('data/redshiftTrain.csv');
testData = importdata('data/redshiftTest.csv');

trainX = trainData(:, 1:10);
testX  = testData(:, 1:10);
trainY = trainData(:, 11);
testY = testData(:, 11);


%Number of trees in the forest
B = [100, 200, 300];
% %Minimum number of leaf node observations
% L = [10,  50,  100, 300, 600, 800, 1000];
%number of splits for cross validation
S = 5;
blockDimension = int32(size(trainData,1)/S);

accs = [];

for b = 1:length(B)
    nrTrees = B(b);
%     for l = 1:length(L)
%         minLeaf = L(l);
        %apply cross validation
        sumRMS = 0;
        for s = 1:S
            Test = trainData((s-1)*blockDimension + 1:s*blockDimension,:);
            Train = trainData;
            Train((s-1)*blockDimension + 1:s*blockDimension,:) = [];
            
            rTrees = cell(nrTrees,1);
            %bootstrap sampling
            for i = 1:nrTrees
                samples = zeros(size(Train));
                for j = 1:size(samples,1)
                    r = randi([1,size(samples,1)]);
                    samples(j,:) = Train(r,:);
                end
                
                samplesX = samples(:,1:10);
                samplesY = samples(:,11);
                
                
                rtree = fitrtree(samplesX,samplesY, 'Prune', 'off');
                rTrees{i,1} = rtree;
            end
            
            %calculate predictions for each sample in the validation block
            testX = Test(:, 1:10);
            testY = Test(:, 11);
            
            predictions = zeros(size(testY));
            for i = 1:size(predictions,1)
                predictionCount = 0;
                for j = 1:nrTrees
                   
                    predictionCount = predictionCount + predict(rTrees{j},testX(i,:));
                end
                predictions(i,1) = predictionCount/nrTrees;
                
            end
            
            %calculate RMS for the validation block
            sumRMS = sumRMS + my_ms(predictions,testY);
            
        end
        %calculate average RMS for the current cross validation
        accs = [nrTrees sumRMS/S; accs];
        disp(nrTrees);
        disp(my_ms(predictions,testY));
        disp('------------------');
        
 %  end
end

disp(accs);

bestNrTrees = find(accs == min(accs(:)));
%build the random forest model with the parameters that were found

%bootstrap sampling
rTreesModel = cell(bestNrTrees,1);
for i = 1:bestNrTrees
    samples = zeros(size(trainData));
    for j = 1:size(samples,1)
       r = randi([1,size(samples,1)]);
       samples(j,:) = trainData(r,:);
    end
                
    samplesX = samples(:,1:10);
    samplesY = samples(:,11);
                
    rtree = fitrtree(samplesX,samplesY, 'Prune', 'off');
    rTrees{i,1} = rtree;
end
            
            
            
%apply prediction for train data into each tree
%calculate predictions for each sample in the train set
predictions = zeros(size(trainY));
for i = 1:size(predictions,1)
    predictionCount = 0;
    for j = 1:bestNrTrees
        predictionCount = predictionCount + predict(rTrees{j},trainX(i,:));
    end
    predictions(i,1) = predictionCount/bestNrTrees;
                
end

trainDataRMS = my_ms(predictions,trainY);
disp('mean-squared-error for the train set: ')
disp(trainDataRMS)

%apply prediction for test data into each tree
%calculate predictions for each sample in the test set
predictions = zeros(size(testY));
for i = 1:size(predictions,1)
   predictionCount = 0;
   for j = 1:bestNrTrees
   predictionCount = predictionCount + predict(rTrees{j},testX(i,:));
   end
   predictions(i,1) = predictionCount/bestNrTrees;
end

  testDataRMS = my_ms(predictions,testY);
  disp('mean-squared-error for the test set: ')
  disp(testDataRMS)


%%
%Question 3 (binary classification) 
%linear method : linear SVM
clearvars();
fprintf('Question 3 - linear binary classifcation using SVM\n');
%read and normalize data
trainData = importdata('data/keystrokesTrainTwoClass.csv');
testData = importdata('data/keystrokesTestTwoClass.csv');

Means = mean(trainData(:,1:21), 1);
Stds = std(trainData(:,1:21), 1, 1);

normalizedTrainData = my_normalize(trainData, Means, Stds,1,21,22);
normalizedTestData = my_normalize(testData, Means, Stds,1,21,22);

%shuffle the train data so I get almost same number of patterns with label 1 and 0 into the model creation
%during cross validation for each split
normalizedShuffledTrainData= normalizedTrainData(randperm(size(normalizedTrainData,1)),:);

trainX = normalizedShuffledTrainData(:, 1:21);
testX  = normalizedTestData(:, 1:21);
trainY = normalizedShuffledTrainData(:, 22);
testY = normalizedTestData(:, 22);

%mean and standard deviation for test data to check the results of
%normalization with mean and std from train data
testMeansNorm = mean(normalizedTestData(:,1:21), 1);
testStdsNorm = std(normalizedTestData(:,1:21), 1, 1);


%regularization parameters
Cs = [0.01, 0.1, 1, 10, 100, 1000, 10000];

%best C and gamma for train data
blockDimensions = [128,128,128,128,128];
bestCNorm = my_fiveFoldCV(trainX, trainY, Cs,blockDimensions);

%train the model using the best C and gamma
modelNorm = svmtrain(trainX, trainY, 'kernel_function','linear','boxconstraint',bestCNorm,'autoscale',false);

%classify train and test data
trainPredYNorm = svmclassify(modelNorm,trainX);
testPredYNorm = svmclassify(modelNorm,testX);

%calculate accuracy
trainAccuracyNorm = 1 - (nnz(trainPredYNorm - trainY)) / length(trainPredYNorm);
testAccuracyNorm = 1 - (nnz(testPredYNorm - testY)) / length(testPredYNorm);

fprintf('train accuracy:');
disp(trainAccuracyNorm);
fprintf('test accuracy:');
disp(testAccuracyNorm);

%calcualte sensitivity and specifity for train set

%count number of patterns correctly classified as belonging to positiv
%class for sensitivity and for specifity count the number of patterns
%correctly classified as belonging to negative class
sumTrainPredYNormTrainY = trainPredYNorm + trainY;
sensitiviyTrain = size(find(sumTrainPredYNormTrainY(:) == 2),1);
specifityTrain = size(find(sumTrainPredYNormTrainY(:) == 0),1);
%divide by number of positive or negative patterns in train and divide
%to obtain final value for specifity and sensitivity
nrPositiveTrain = size(find(trainY(:,1) == 1),1);
nrNegativeTrain = size(find(trainY(:,1) == 0),1);
sensitiviyTrain = sensitiviyTrain/nrPositiveTrain;
specifityTrain = specifityTrain/nrNegativeTrain;

fprintf('train sensitivity:');
disp(sensitiviyTrain);
fprintf('train specifity:');
disp(specifityTrain);


%calcualte sensitivity and specifity for test set

%count number of patterns correctly classified as belonging to positiv
%class for sensitivity and for specifity count the number of patterns
%correctly classified as belonging to negative class
sumTestPredYNormTestY = testPredYNorm + testY;
sensitiviyTest = size(find(sumTestPredYNormTestY(:) == 2),1);
specifityTest = size(find(sumTestPredYNormTestY(:) == 0),1);

%divide by number of positive or negative patterns in train and divide
%to obtain final value for specifity and sensitivity
nrPositiveTest = size(find(testY(:,1) == 1),1);
nrNegativeTest = size(find(testY(:,1) == 0),1);

sensitiviyTest = sensitiviyTest/nrPositiveTest;
specifityTest = specifityTest/nrNegativeTest;

fprintf('test sensitivity:');
disp(sensitiviyTest);
fprintf('test specifity:');
disp(specifityTest);


%%
%Question 3 (binary classification) 
%linear method : SVM with non-linear kernel
clearvars();
fprintf('Question 3 - non-linear binary classifcation using SVM\n');
%read and normalize data
trainData = importdata('data/keystrokesTrainTwoClass.csv');
testData = importdata('data/keystrokesTestTwoClass.csv');

Means = mean(trainData(:,1:21), 1);
Stds = std(trainData(:,1:21), 1, 1);

normalizedTrainData = my_normalize(trainData, Means, Stds,1,21,22);
normalizedTestData = my_normalize(testData, Means, Stds,1,21,22);

%shuffle the train data so I get almost same number of patterns with label 1 and 0 into the model creation
%during cross validation for each split
normalizedShuffledTrainData= normalizedTrainData(randperm(size(normalizedTrainData,1)),:);


trainX = normalizedShuffledTrainData(:, 1:21);
testX  = normalizedTestData(:, 1:21);
trainY = normalizedShuffledTrainData(:, 22);
testY = normalizedTestData(:, 22);

%mean and standard deviation for test data to check the results of
%normalization with mean and std from train data
testMeansNorm = mean(normalizedTestData(:,1:21), 1);
testStdsNorm = std(normalizedTestData(:,1:21), 1, 1);


%regularization parameters
Cs = [0.01, 0.1, 1, 10, 100, 1000, 10000];
%kernel parameters
gammas = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100];

%best C and gamma for non normalizaed data
blockDimensions = [128,128,128,128,128];
[bestCNorm, bestGammaNorm] = my_fiveFoldGaussianKernelCV(trainX, trainY, Cs, gammas,blockDimensions);

%train the model using the best C and gamma and gaussian kernel
kernel = @(x,y) my_kernel(x, y, bestGammaNorm);
model = svmtrain(trainX, trainY, ...
                'autoscale',false, ... 
                'boxconstraint',ones(size(trainX,1),1) * bestCNorm, ...
                'kernel_function',kernel);

%classify train and test data
trainPredYNorm = svmclassify(model,trainX);
testPredYNorm = svmclassify(model,testX);

trainAccuracyNorm = 1 - (nnz(trainPredYNorm - trainY)) / length(trainPredYNorm);
testAccuracyNorm = 1 - (nnz(testPredYNorm - testY)) / length(testPredYNorm);

fprintf('train accuracy:');
disp(trainAccuracyNorm);
fprintf('test accuracy:');
disp(testAccuracyNorm);

%calcualte sensitivity and specifity for train set

%count number of patterns correctly classified as belonging to positiv
%class for sensitivity and for specifity count the number of patterns
%correctly classified as belonging to negative class
sumTrainPredYNormTrainY = trainPredYNorm + trainY;
sensitiviyTrain = size(find(sumTrainPredYNormTrainY(:) == 2),1);
specifityTrain = size(find(sumTrainPredYNormTrainY(:) == 0),1);
%divide by number of positive or negative patterns in train and divide
%to obtain final value for specifity and sensitivity
nrPositiveTrain = size(find(trainY(:,1) == 1),1);
nrNegativeTrain = size(find(trainY(:,1) == 0),1);
sensitiviyTrain = sensitiviyTrain/nrPositiveTrain;
specifityTrain = specifityTrain/nrNegativeTrain;

fprintf('train sensitivity:');
disp(sensitiviyTrain);
fprintf('train specifity:');
disp(specifityTrain);


%calcualte sensitivity and specifity for test set

%count number of patterns correctly classified as belonging to positiv
%class for sensitivity and for specifity count the number of patterns
%correctly classified as belonging to negative class
sumTestPredYNormTestY = testPredYNorm + testY;
sensitiviyTest = size(find(sumTestPredYNormTestY(:) == 2),1);
specifityTest = size(find(sumTestPredYNormTestY(:) == 0),1);

%divide by number of positive or negative patterns in train and divide
%to obtain final value for specifity and sensitivity
nrPositiveTest = size(find(testY(:,1) == 1),1);
nrNegativeTest = size(find(testY(:,1) == 0),1);

sensitiviyTest = sensitiviyTest/nrPositiveTest;
specifityTest = specifityTest/nrNegativeTest;

fprintf('test sensitivity:');
disp(sensitiviyTest);
fprintf('test specifity:');
disp(specifityTest);


%%
%Question 4 (principal component analysis) 
clearvars();
fprintf('Question 4 - Principal Component Analysis\n');
%read  data
trainData = importdata('data/keystrokesTrainTwoClass.csv');

Means = mean(trainData(:,1:21), 1);
Stds = std(trainData(:,1:21), 1, 1);

trainData = my_normalize(trainData, Means, Stds,1,21,22);
trainX = trainData(:, 1:21);


%calcualte mean
s = sum(trainX);
m = s/size(trainX,1);

%calculate scatter matrix
scatter = sum((trainX - repmat(m,size(trainX,1),1))*transpose(trainX - repmat(m,size(trainX,1),1)));

%calculate covariance
covariance = cov(trainX);

%calculate eigenvectors and eigenvalues
[eigenvec, eigenval] = eig(covariance);


%plot the eigenspectrum
eigenspectrum = sort(diag(eigenval), 'descend');
plot(1:21,eigenspectrum),title('Plot of the eigenspectrum'),xlabel('Principal Component'),ylabel('Eigenvalue');



%sorting the eigenvectors by decreasing eigenvalues
[sortedEigenvec,sortedEigenval] = eigs(covariance,2);

%scatter plot of the data projected on the first two principal components
figure;
transformedXMatrix = transpose(transpose(sortedEigenvec)*transpose(trainX));
axis([-3 3 -3 3])


plot(transformedXMatrix(:,1),transformedXMatrix(:,2),'bO'),title('Scatter plot of the data projected on the first two principal components');


%%
%Question 5 (clustering)
clearvars();
fprintf('Question 5 - Clustering\n');
%read and normalize data
trainData = importdata('data/keystrokesTrainTwoClass.csv');

Means = mean(trainData(:,1:21), 1);
Stds = std(trainData(:,1:21), 1, 1);

trainData = my_normalize(trainData, Means, Stds,1,21,22);
trainData= trainData(randperm(size(trainData,1)),:);

trainX = trainData(:, 1:21);

%create the initial centroids: take the first 2 patterns from shuffled
%train data

centroid1 = trainX(1,:);
centroid2 = trainX(2,:);


centroid1Prev = zeros(size(trainX(1,:)));
centroid2Prev = zeros(size(trainX(1,:)));

cluster1 = [];
cluster2 = [];


%
for i = 1 : 100
    %create initial empty clusters and assign to each cluster the patterns
    %that are the closest (by calculating Euclidian distance)
    cluster1 = [];
    cluster2 = [];
    for p = 1: size(trainX,1)
        pattern = trainX(p,:);
        d1 = sqrt(sum(centroid1 - pattern).^ 2);
        d2 = sqrt(sum(centroid2 - pattern).^ 2);
        
        if(d1<d2)
            cluster1 = [pattern ; cluster1];
        else
            cluster2 = [pattern ; cluster2];
        end
    end
   
    %calcualte mean for each cluster and find next centroids
    m1 = mean(cluster1);
    m2 = mean(cluster2);
    
    distances1 = sqrt(sum((cluster1 - ones(size(cluster1))*diag(m1)),2).^ 2);
    distances2 = sqrt(sum((cluster2 - ones(size(cluster2))*diag(m2)),2).^ 2);
    
    c1 = find(distances1 == min(distances1(:)));
    c2 = find(distances2 == min(distances2(:)));
    
    centroid1 = cluster1(c1(1),:);
    centroid2 = cluster2(c2(1),:);
    
    %check the stop condition
    if(sum(centroid1 == centroid1Prev) && sum(centroid2 == centroid2Prev))
        disp('stop at iteration');
        disp(i);
        break;
    end
    
    centroid1Prev = centroid1;
    centroid2Prev = centroid2;
 
end

%transform points in clusters and centroids to have only 2 coordinates
covariance = cov(trainX);
[sortedEigenvec,sortedEigenval] = eigs(covariance,2);
transformedCluster1 = transpose(transpose(sortedEigenvec)*transpose(cluster1));
transformedCluster2 = transpose(transpose(sortedEigenvec)*transpose(cluster2));
transformedCentroid1 = transpose(transpose(sortedEigenvec)*transpose(centroid1));
transformedCentroid2 = transpose(transpose(sortedEigenvec)*transpose(centroid2));

%plot the first 2 parameters of patterns and the final centroids 
plot(transformedCluster1(:,1),transformedCluster1(:,2),'b.')
hold on; plot(transformedCluster2(:,1),transformedCluster2(:,2),'r.');hold off;
hold on; plot(transformedCentroid1(1),transformedCentroid1(2),'g*');hold off;
hold on; plot(transformedCentroid2(1),transformedCentroid2(2),'y*');hold off;
legend('cluster 1','cluster 2','centroid 1', 'centroid 2');

%%
%Question 6 (linear using LDA)
clearvars();
fprintf('Question 6 - multi-class cassification using LDA\n');
trainData = importdata('data/keystrokesTrainMulti.csv');
testData = importdata('data/keystrokesTestMulti.csv');

%calcualte train accuracy and test accuracy for both test and train data
[trainAcc, testAcc] = my_lda(trainData, testData);

%calculate and display error
trainError = 1 - trainAcc;
testError = 1 - testAcc;

disp('Train error');
disp (trainError);
disp('Test error');
disp (testError);


%%
%Question 6 (nonlinear using knn)
clearvars();
fprintf('Question 6 - multi-class cassification using k-nearest neighbor\n');
%read and normalize data
trainData = importdata('data/keystrokesTrainMulti.csv');
testData = importdata('data/keystrokesTestMulti.csv');

Means = mean(trainData(:,1:21), 1);
Stds = std(trainData(:,1:21), 1, 1);

normalizedTrainData = my_normalize(trainData, Means, Stds,1,21,22);
normalizedTestData = my_normalize(testData, Means, Stds,1,21,22);


%shuffle the train data so I get almost same number of patterns with each label 
%when doing the cross validation for each split
normalizedShuffledTrainData= normalizedTrainData(randperm(size(normalizedTrainData,1)),:);


%%apply 5-fold cross validation
S = 5;
kRiskMatrix = [];
blockDimension = int32(size(trainData,1)/S);

for k = 1:2:25
   kAverage = 0;
   for s = 1:S
        %split data and apply knn
        Test = normalizedShuffledTrainData((s-1)*blockDimension + 1:s*blockDimension,:);
        Train = normalizedShuffledTrainData;
        Train((s-1)*blockDimension + 1:s*blockDimension,:) = [];
        kAverage = kAverage + kNN(k,Train, Test); 
   end
   new_row =[k kAverage/5] ; 
   kRiskMatrix = [kRiskMatrix ; new_row];
end
%find the best k
kBestIndex = find(kRiskMatrix(2,:) == min(kRiskMatrix(2,:)));
kBest = kRiskMatrix(kBestIndex,1);
%calculate error for train and test data
eNewRiskTrain = kNN(kBest,normalizedTrainData,normalizedTrainData);
eNewRiskTest = kNN(kBest,normalizedTrainData,normalizedTestData);
   
disp('Train error');
disp (eNewRiskTrain);
disp('Test error');
disp (eNewRiskTest);




