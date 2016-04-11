function [trainAccuracy, testAccuracy] = my_lda( trainData, testData )
%MY_LDA Summary of this function goes here
%   Detailed explanation goes here

%get first 21 columns of the train data
xTrain = trainData(:,1:21);
%get the last column of classification
yTrain = trainData(:,22);

%get first 2 columns of the test data
xTest = testData(:,1:21);
%get the last column of classification
yTest = testData(:,22);

%get indexes
class1_indexes = find(yTrain == 1);
class2_indexes = find(yTrain == 2);
class3_indexes = find(yTrain == 3);
class4_indexes = find(yTrain == 4);
class5_indexes = find(yTrain == 5);

%number of elements of each class
l_k = [length(class1_indexes);
       length(class2_indexes);
       length(class3_indexes);
       length(class4_indexes);
       length(class5_indexes)];

%calulate priors
priors = l_k ./ length(yTrain);

%separate into classes
X_1 = xTrain (class1_indexes,:);
X_2 = xTrain (class2_indexes,:);
X_3 = xTrain (class3_indexes,:);
X_4 = xTrain (class4_indexes,:);
X_5 = xTrain (class5_indexes,:);

%calculate means for each class
mu1 = 1 / l_k(1) * sum(X_1);
mu2 = 1 / l_k(2) * sum(X_2);
mu3 = 1 / l_k(3) * sum(X_3);
mu4 = 1 / l_k(4) * sum(X_4);
mu5 = 1 / l_k(5) * sum(X_5);


%calculate covariance for each class
cov1 = zeros(21,21);
for i=1:length(class1_indexes)
    index = class1_indexes(i);
    v = xTrain(index,:) - mu1;
    cov1 = cov1 + (v' * v);
end

cov2 = zeros(21,21);
for i=1:length(class2_indexes)
    index = class2_indexes(i);
    v = xTrain(index,:) - mu2;
    cov2 = cov2 + (v' * v);
end

cov3 = zeros(21,21);
for i=1:length(class1_indexes)
    index = class3_indexes(i);
    v = xTrain(index,:) - mu3;
    cov3 = cov3 + (v' * v);
end

cov4 = zeros(21,21);
for i=1:length(class4_indexes)
    index = class4_indexes(i);
    v = xTrain(index,:) - mu4;
    cov4 = cov4 + (v' * v);
end

cov5 = zeros(21,21);
for i=1:length(class5_indexes)
    index = class5_indexes(i);
    v = xTrain(index,:) - mu5;
    cov5 = cov5 + (v' * v);
end



covariance = (cov1 + cov2 + cov3 + cov4 + cov5 )/ (length(trainData)-5); 
% disp(covariance);
% disp(cov(X_0) + cov(X_1) + cov(X_2));

% prediction matrix for train data
predTrain = zeros(length(xTrain),1);
for i=1:length(xTrain)
    x = xTrain(i,:);
    class1 = my_linearclassifier(x,covariance,mu1,priors(1));
    class2 = my_linearclassifier(x,covariance,mu2,priors(2));
    class3 = my_linearclassifier(x,covariance,mu3,priors(3));
    class4 = my_linearclassifier(x,covariance,mu4,priors(4));
    class5 = my_linearclassifier(x,covariance,mu5,priors(5));
    pred = [class1; class2; class3; class4; class5];
    [~,index] = max(pred);
    predTrain(i) = index;
end

%calculate accuracy for train data
trainAccuracy    = 1 - (nnz(predTrain - yTrain) / length(yTrain));

% prediction matrix for test data
predTest = zeros(length(xTest),1);
for i=1:length(xTest)
    x = xTest(i,:);
    class1 = my_linearclassifier(x,covariance,mu1,priors(1));
    class2 = my_linearclassifier(x,covariance,mu2,priors(2));
    class3 = my_linearclassifier(x,covariance,mu3,priors(3));
    class4 = my_linearclassifier(x,covariance,mu4,priors(4));
    class5 = my_linearclassifier(x,covariance,mu5,priors(5));
    pred = [class1; class2; class3; class4; class5];
    [~,index] = max(pred);
    predTest(i) = index;
end

%calculate accuracy for test data
testAccuracy = 1 - (nnz(predTest - yTest) / length(yTest));

% %testing out our work
%  linclass = ClassificationDiscriminant.fit(xTrain,yTrain);
%  pred_train = predict(linclass,xTrain);
%  pred_test = predict(linclass,xTest);
%  
%  theirAccTrain = 1 - (nnz(pred_train - yTrain) / length(yTrain));
%  theirAccTest = 1 - (nnz(pred_test - yTest) / length(yTest));
%  
%  disp(theirAccTrain);
%  disp(theirAccTest);

end

