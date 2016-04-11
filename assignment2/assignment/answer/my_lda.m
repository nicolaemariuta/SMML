function [trainAccuracy, testAccuracy] = my_lda( trainData, testData )
%MY_LDA Summary of this function goes here
%   Detailed explanation goes here

%get first 2 columns of the train data
xTrain = trainData(:,1:2);
%get the last column of classification
yTrain = trainData(:,3);

%get first 2 columns of the test data
xTest = testData(:,1:2);
%get the last column of classification
yTest = testData(:,3);

%get indexes
class0_indexes = find(yTrain == 0);
class1_indexes = find(yTrain == 1);
class2_indexes = find(yTrain == 2);

%number of elements of each class
l_k = [length(class0_indexes);
       length(class1_indexes);
       length(class2_indexes)];

%calulate priors
priors = l_k ./ length(yTrain);

%separate into classes
X_0 = xTrain (class0_indexes,:);
X_1 = xTrain (class1_indexes,:);
X_2 = xTrain (class2_indexes,:);

%calculate means for each class
mu0 = 1 / l_k(1) * sum(X_0);
mu1 = 1 / l_k(2) * sum(X_1);
mu2 = 1 / l_k(3) * sum(X_2);

%calculate covariance for each class
cov0 = zeros(2,2);
for i=1:length(class0_indexes)
    index = class0_indexes(i);
    v = xTrain(index,:) - mu0;
    cov0 = cov0 + (v' * v);
end
cov0 = cov0 ./ l_k(1);

cov1 = zeros(2,2);
for i=1:length(class1_indexes)
    index = class1_indexes(i);
    v = xTrain(index,:) - mu1;
    cov1 = cov1 + (v' * v);
end
cov1 = cov1 ./ l_k(2);


cov2 = zeros(2,2);
for i=1:length(class2_indexes)
    index = class2_indexes(i);
    v = xTrain(index,:) - mu2;
    cov2 = cov2 + (v' * v);
end
cov2 = cov2 ./ l_k(3);

covariance = cov0 + cov1 + cov2;
% disp(covariance);
% disp(cov(X_0) + cov(X_1) + cov(X_2));

% prediction matrix for train data
predTrain = zeros(length(xTrain),1);
for i=1:length(xTrain)
    x = xTrain(i,:);
    class0 = my_linearclassifier(x,covariance,mu0,priors(1));
    class1 = my_linearclassifier(x,covariance,mu1,priors(2));
    class2 = my_linearclassifier(x,covariance,mu2,priors(3));
    pred = [class0; class1; class2];
    [~,index] = max(pred);
    predTrain(i) = index - 1;
end

%calculate accuracy for train data
trainAccuracy    = 1 - (nnz(predTrain - yTrain) / length(yTrain));

% prediction matrix for test data
predTest = zeros(length(xTest),1);
for i=1:length(xTest)
    x = xTest(i,:);
    class0 = my_linearclassifier(x,covariance,mu0,priors(1));
    class1 = my_linearclassifier(x,covariance,mu1,priors(2));
    class2 = my_linearclassifier(x,covariance,mu2,priors(3));
    pred = [class0; class1; class2];
    [~,index] = max(pred);
    predTest(i) = index - 1;
end

%calculate accuracy for test data
testAccuracy = 1 - (nnz(predTest - yTest) / length(yTest));

%testing out our work
% linclass = ClassificationDiscriminant.fit(xTrain,yTrain);
% pred_train = predict(linclass,xTrain);
% pred_test = predict(linclass,xTest);
% 
% theirAccTrain = 1 - (nnz(pred_train - yTrain) / length(yTrain));
% theirAccTest = 1 - (nnz(pred_test - yTest) / length(yTest));
% 
% disp(theirAccTrain);
% disp(theirAccTest);

end

