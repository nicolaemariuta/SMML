%% Section III.1.1
clearvars;
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
hiddenLayerSize = 20;

inputBiasSize = 1;
outputBiasSize = 1;

w1 = 6*randn(inputLayerSize + inputBiasSize,hiddenLayerSize)-3;
w2 = 6*randn(hiddenLayerSize + outputBiasSize,outputLayerSize)-3;


%forward propagation
[z2,a2,z3,yHat] = forward( xTrain, w1, w2 );
J = costFunction( xTrain, yTrain, w1, w2 );
[dJW1,dJW2] = costFunctionPrime( xTrain,yTrain,w1,w2 );

gradient =  computeGradients(xTrain,yTrain,w1,w2);
gradient2 = computeNumericalGradient(xTrain,yTrain,w1,w2,hiddenLayerSize, ...
    inputLayerSize,inputBiasSize, outputBiasSize, outputLayerSize);

[endErrors, validErrors] = train(xTrain, yTrain, xTest, yTest, w1, w2, ...
    10000, 10^(-8), hiddenLayerSize, inputLayerSize,inputBiasSize, ...
    outputBiasSize, outputLayerSize);

%  [w_1,w_2 ]  = reshapeFunction(gradient,hiddenLayerSize, inputLayerSize, ...
%      inputBiasSize, outputBiasSize,outputLayerSize);

%% Section III.2.1 & III.2.2
clearvars();
%read and normalize data
trainData = importdata('data/parkinsonsTrainStatML.dt');
testData = importdata('data/parkinsonsTestStatML.dt');

trainX = trainData(:, 1:22);
testX  = testData(:, 1:22);
trainY = trainData(:, 23);
testY = testData(:, 23);

Means = mean(trainData(:,1:22), 1);
Stds = std(trainData(:,1:22), 1, 1);

normalizedTrainData = my_normalize(trainData, Means, Stds,1,22,23);
normalizedTestData = my_normalize(testData, Means, Stds,1,22,23);

testMeansNorm = mean(normalizedTestData(:,1:22), 1);
testStdsNorm = std(normalizedTestData(:,1:22), 1, 1);

%regularization parameters
Cs = [0.01, 0.1, 1, 10, 100, 1000, 10000];
%kernel parameters
gammas = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100];

%best C and gamma for non normalizaed data
[bestC, bestGamma] = my_fiveFoldCV(trainX, trainY, Cs, gammas);

normTrainX = normalizedTrainData(:, 1:22);
normTestX  = normalizedTestData(:, 1:22);
normTrainY = normalizedTrainData(:, 23);
normTestY = normalizedTestData(:, 23);

%best C and gamma for normalizaed data
[bestCNorm, bestGammaNorm] = my_fiveFoldCV(normTrainX, normTrainY, Cs, gammas);

%non normalized data
bestSigma = sqrt(1/(2*bestGamma));
model = svmtrain(trainX, trainY, 'kernel_function','rbf','rbf_sigma',bestSigma,'boxconstraint',bestC,'autoscale',false);

trainPredY = svmclassify(model,trainX);
testPredY = svmclassify(model,testX);

trainAccuracy = 1 - (nnz(trainPredY - trainY)) / length(trainPredY);
testAccuracy = 1 - (nnz(testPredY - testY)) / length(testPredY);

%normalized data
bestSigmaNorm = sqrt(1/(2*bestGammaNorm));
modelNorm = svmtrain(normTrainX, normTrainY, 'kernel_function','rbf','rbf_sigma',bestSigmaNorm,'boxconstraint',bestCNorm,'autoscale',false);

trainPredYNorm = svmclassify(modelNorm,normTrainX);
testPredYNorm = svmclassify(modelNorm,normTestX);

trainAccuracyNorm = 1 - (nnz(trainPredYNorm - normTrainY)) / length(trainPredYNorm);
testAccuracyNorm = 1 - (nnz(testPredYNorm - normTestY)) / length(testPredYNorm);

%% Section III.2.3
clearvars();
%read and normalize data
trainData = importdata('data/parkinsonsTrainStatML.dt');
testData = importdata('data/parkinsonsTestStatML.dt');

Means = mean(trainData(:,1:22), 1);
Stds = std(trainData(:,1:22), 0, 1);

normalizedTrainData = my_normalize(trainData, Means, Stds,1,22,23);
normalizedTestData = my_normalize(testData, Means, Stds,1,22,23);

trainX = normalizedTrainData(:, 1:22);
testX  = normalizedTestData(:, 1:22);
trainY = normalizedTrainData(:, 23);
testY = normalizedTestData(:, 23);

gamma = 0.01; 

kernel = @(x,y) my_kernel(x, y, gamma);

sigma = sqrt(1/(2*gamma));
Cs = [0.01, 0.1, 1, 10, 100, 1000, 10000];

%number of free and bounded support vectors
nrBoundedXs = zeros(1,length(Cs));
nrFreeXs = zeros(1,length(Cs));

for i = 1:length(Cs)
    model = svmtrain(trainX, trainY, ...
        'autoscale',false, ... 
        'boxconstraint',ones(size(trainX,1),1) * Cs(i), ...
        'kernel_function',kernel);
   
    N  = size(model.SupportVectorIndices, 1);
   
    nrBoundedXs(1,i) = sum(abs(model.Alpha) == Cs(i));
    nrFreeXs(1,i) = N-sum(abs(model.Alpha) == Cs(i));
end

disp(nrBoundedXs);
disp(nrFreeXs);


