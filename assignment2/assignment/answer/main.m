%%
% II.1.1
%
clear;fprintf('- II.1.1 -\n');

trainData = importdata('data/IrisTrain2014.dt');
testData = importdata('data/IrisTest2014.dt');

[trainAcc, testAcc] = my_lda(trainData, testData);

trainError = 1 - trainAcc;
testError = 1 - testAcc;

disp (trainError);
disp (testError);

%%
% II.1.2
%
clear;fprintf('- II.1.2 -\n');

trainData = importdata('data/IrisTrain2014.dt');
testData = importdata('data/IrisTest2014.dt');

Means = mean(trainData(:,1:2), 1);
Stds = std(trainData(:,1:2), 0, 1);

normTrainData = my_normalize(trainData, Means, Stds);
normMeans = mean(normTrainData(:,1:2), 1);
normVars = var(normTrainData(:,1:2), 0, 1);

normTestData = my_normalize(testData, Means, Stds);

[trainAcc, testAcc] = my_lda(normTrainData, normTestData);

trainError = 1 - trainAcc;
testError = 1 - testAcc;

disp (trainError );
disp (testError);

%%
% II.2.1
%
clear;fprintf('- II.2.1 -\n');

trainData = importdata('data/sunspotsTrainStatML.dt');
testData = importdata('data/sunspotsTestStatML.dt');
l = length(trainData);
testYears = 1916:2011;

%Selection 1
Phi1 = horzcat(ones(l,1),trainData(:,3),trainData(:,4));
wML1 = my_wML(Phi1, trainData(:,6));
X1 = testData (:,3:4);
prediction1 = my_lmfunction(wML1,X1);

%RMS 1
rms1 = my_rms(prediction1,testData(:,6));

figure;
hold on;
title('Selection 1 - measured(blue) vs. predicted(red) 1916 - 2011','FontSize', 15);
plot(testYears, prediction1, 'r', testYears, testData(:,6), 'b');
legend('Actual sunspots', 'Predicted sunspots');
hold off;

%Selection 2
Phi2 = horzcat(ones(l,1),trainData(:,5));
wML2 = my_wML(Phi2, trainData(:,6));
X2 = testData (:,5);
prediction2 = my_lmfunction(wML2,X2);

%RMS 2
rms2 = my_rms(prediction2,testData(:,6));

%training and test set with predicted target variables plot
figure;
hold on;
title('Selection 2 - training and test set with predicted target variables.','FontSize', 15);
plot(trainData(:,5),trainData(:,6),'rx', testData(:,5), testData(:,6), 'gx', testData(:,5), prediction2, 'bx');
legend('Training data', 'Real targets', 'Predicted targets');
hold off;

figure;
hold on;
title('Selection 2 - measured(blue) vs. predicted(red) 1916 - 2011','FontSize', 15);
plot(testYears, prediction2, 'r', testYears, testData(:,6), 'b');
legend('Actual sunspots', 'Predicted sunspots');
hold off;

%Selection 3
Phi3 = horzcat(ones(l,1),trainData(:,1),trainData(:,2),trainData(:,3),trainData(:,4),trainData(:,5));
wML3 = my_wML(Phi3, trainData(:,6));
X3 = testData (:,1:5);
prediction3 = my_lmfunction(wML3,X3);

%RMS 1
rms3 = my_rms(prediction3,testData(:,6));

figure;
hold on;
title('Selection 3 - measured(blue) vs. predicted(red) 1916 - 2011','FontSize', 15);
plot(testYears, prediction3, 'r', testYears, testData(:,6), 'b');
legend('Actual sunspots', 'Predicted sunspots');
hold off;

%%
% II.2.2
%
clear;fprintf('- II.2.2 -\n');

trainData = importdata('data/sunspotsTrainStatML.dt');
testData = importdata('data/sunspotsTestStatML.dt');
l = length(trainData);
testYears = 1916:2011;

%Phis
Phi1 = horzcat(ones(l,1),trainData(:,3),trainData(:,4));
Phi2 = horzcat(ones(l,1),trainData(:,5));
Phi3 = horzcat(ones(l,1),trainData(:,1),trainData(:,2),trainData(:,3),trainData(:,4),trainData(:,5));

%Previous RMS
rms1_ML = 35.4651;
rms2_ML = 28.8398;
rms3_ML = 18.77;

%choose alpha array
alphas = 10.^(-10:1:10);
rms1 = zeros(length(alphas),1);
rms2 = zeros(length(alphas),1);
rms3 = zeros(length(alphas),1);

for i=1:length(alphas)
    alpha = alphas(i);
    
    %wMAPs
    wMAP1 = my_wMAP(trainData(:,3:4), trainData(:,6), Phi1, alpha, 1);
    wMAP2 = my_wMAP(trainData(:,5), trainData(:,6), Phi2, alpha, 1);
    wMAP3 = my_wMAP(trainData(:,1:5), trainData(:,6), Phi3, alpha, 1);
    
    %predictions
    prediction1 = my_lmfunction(wMAP1,testData(:,3:4));
    prediction2 = my_lmfunction(wMAP2,testData(:,5));
    prediction3 = my_lmfunction(wMAP3,testData(:,1:5));
    
    %calcualte RMSs
    rms1(i) = my_rms( prediction1, testData(:, 6));
    rms2(i) = my_rms( prediction2, testData(:, 6));
    rms3(i) = my_rms( prediction3, testData(:, 6));
   
end

figure;
title('RMS error plotted over alpha values between 10^-10 and 10^10','FontSize',15);
semilogx(10.^(-10:1:10), rms1, 'r', 'LineWidth', 2);
hold on;
semilogx(10.^(-10:1:10), rms2, 'g', 'LineWidth', 2);
hold off;
hold on
semilogx(10.^(-10:1:10), rms3, 'b', 'LineWidth', 2);
hold off;
legend('Selection 1', 'Selection 3', 'Selection 3');
