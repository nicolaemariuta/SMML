%1.1 Neural network implementation
clear;fprintf('- III.1.1 -\n');

%read and normalize data
trainData = importdata('data/sincTrain25.dt');
testData = importdata('data/sincValidate10.dt');

xTrain = trainData(:,1);
yTrain = trainData(:,2);

xTest = testData(:,1);
yTest = testData(:,2);


%weights
inputLayerSize = 1;
outputLayerSize = 1;
hiddenLayerSize = 3;

inputBiasSize = 1;
outBiasSize = 1;

w1 = 6*randn(inputLayerSize,hiddenLayerSize)-3;
w2 = 6*randn(hiddenLayerSize,outputLayerSize)-3;

%forward propagation
[dJW1,dJW2] = costFunctionPrime(xTrain,yTrain,w1,w2);
gradient =  computeGradients(xTrain,yTrain,w1,w2);
gradient2 = computeNumericalGradient(xTrain,yTrain,w1,w2);


%training


%%
%data normalization
trainData = importdata('data/parkinsonsTrainStatML.dt');
testData = importdata('data/parkinsonsTestStatML.dt');

Means = mean(trainData(:,1:22), 1);
Stds = std(trainData(:,1:22), 0, 1);

normalizedTrainData = my_normalize(trainData, Means, Stds);
normalizedTestData = my_normalize(testData, Means, Stds);
















