
%Exercise 2.1
[Ox1 , Oy1] = gaussian1d(-1,1);
[Ox2 , Oy2] = gaussian1d(0,2);
[Ox3 , Oy3] = gaussian1d(2,3);

plot(Ox1, Oy1,'r');
hold on; plot(Ox2, Oy2,'b'); hold off;
hold on; plot(Ox3, Oy3,'g'); hold off;
legend('(mu, sigma) = (-1,1)','(mu, sigma) = (0,2)','(mu, sigma) = (2,3)')

%%
%Exercise 2.2
%S = randn(100,2);
%plot(S);
figure;
mu = [1 2];
sigma = [0.3 0.2; 0.2 0.2];
R = chol(sigma,'lower');
z = repmat(mu',1,100) + R*randn(100,2)';
z=z';

plot(z(:,1),z(:,2),'b.');

%Exercise 2.3
%calcualte mean

s = sum(z);
m = s/size(z,1);

hold on; plot(m(:,1),m(:,2),'r.');hold off;
hold on; plot(mu(:,1),mu(:,2),'g.');hold off;


%Exercise 2.4
%calculate covariance
covariance = cov(z);

myCovariance = zeros(2,2);

for i=1:size(z,1)
    xn = z(i,:);
    myCovariance = myCovariance + (xn-m).'*(xn-m);
end

myCovariance = myCovariance/size(z,1);

%covariance=sigma;
[eigenvec, eigenval] = eig(myCovariance);
k = sqrt(diag(eigenval));
axis([-1 4 -1 4])
hold on; plot([m(1) m(1)+k(1)*eigenvec(1,1)],[m(2) m(2)+k(1)*eigenvec(2,1)],'r-');hold off;
hold on; plot([m(1) m(1)+k(2)*eigenvec(1,2)],[m(2) m(2)+k(2)*eigenvec(2,2)],'r-');hold off;


%rotation
alpha = 0;
if eigenval(1,1)<= eigenval(2,2)
    alpha = atan2(eigenvec(2,2),eigenvec(1,2));
else
    alpha = atan2(eigenvec(2,1),eigenvec(1,1));
end

%%
%Rotated Gaussian distributions
figure;
mu = [1 2];
sigma = [0.3 0.2; 0.2 0.2];

rad30 = 30*pi/180;
rad60 = 60*pi/180;
rad90 = 90*pi/180;
radA = alpha;

r30 = [cos(rad30) -sin(rad30); sin(rad30) cos(rad30)];
r60 = [cos(rad60) -sin(rad60); sin(rad60) cos(rad60)];
r90 = [cos(rad90) -sin(rad90); sin(rad90) cos(rad90)];
rA = [cos(radA) -sin(radA); sin(radA) cos(radA)];
sigma30 = r30' * sigma *r30 ;
sigma60 = r60' * sigma *r60 ;
sigma90 = r90' * sigma *r90 ;
sigmaA = rA' * sigma *rA ;


R30 = chol(sigma30,'lower');
R60 = chol(sigma60,'lower');
R90 = chol(sigma90,'lower');
RA = chol(sigmaA,'lower');

z30 = repmat(mu',1,100) + R30*randn(100,2)';
z60 = repmat(mu',1,100) + R60*randn(100,2)';
z90 = repmat(mu',1,100) + R90*randn(100,2)';
zA = repmat(mu',1,100) + RA*randn(100,2)';

z30=z30';
z60=z60';
z90=z90';
zA = zA';

plot(z30(:,1),z30(:,2),'b.');
hold on;plot(z60(:,1),z60(:,2),'y.');hold off;
hold on;plot(z90(:,1),z90(:,2),'r.');hold off;
hold on;plot(zA(:,1),zA(:,2),'g.');hold off;



%Exercise 2.4
%calculate covariance
covariance30 = cov(z30);
covariance60 = cov(z60);
covariance90 = cov(z90);
covarianceA = cov(zA);
%covariance=sigma;
[eigenvec30, eigenval30] = eig(covariance30);
[eigenvec60, eigenval60] = eig(covariance60);
[eigenvec90, eigenval90] = eig(covariance90);
[eigenvecA, eigenvalA] = eig(covarianceA);

k30 = sqrt(diag(eigenval30));
k60 = sqrt(diag(eigenval60));
k90 = sqrt(diag(eigenval90));
kA = sqrt(diag(eigenvalA));

axis([-1 4 -1 4])
hold on; plot([m(1) m(1)+k30(1)*eigenvec30(1,1)],[m(2) m(2)+k30(1)*eigenvec30(2,1)],'b-');hold off;
hold on; plot([m(1) m(1)+k30(2)*eigenvec30(1,2)],[m(2) m(2)+k30(2)*eigenvec30(2,2)],'b-');hold off;

axis([-1 4 -1 4])
hold on; plot([m(1) m(1)+k60(1)*eigenvec60(1,1)],[m(2) m(2)+k60(1)*eigenvec60(2,1)],'y-');hold off;
hold on; plot([m(1) m(1)+k60(2)*eigenvec60(1,2)],[m(2) m(2)+k60(2)*eigenvec60(2,2)],'y-');hold off;

axis([-1 4 -1 4])
hold on; plot([m(1) m(1)+k90(1)*eigenvec90(1,1)],[m(2) m(2)+k90(1)*eigenvec90(2,1)],'r-');hold off;
hold on; plot([m(1) m(1)+k90(2)*eigenvec90(1,2)],[m(2) m(2)+k90(2)*eigenvec90(2,2)],'r-');hold off;

axis([-1 4 -1 4])
hold on; plot([m(1) m(1)+kA(1)*eigenvecA(1,1)],[m(2) m(2)+kA(1)*eigenvecA(2,1)],'g-');hold off;
hold on; plot([m(1) m(1)+kA(2)*eigenvecA(1,2)],[m(2) m(2)+kA(2)*eigenvecA(2,2)],'g-');hold off;


legend('angle = 30','angle = 60','angle = 90', 'angle = alpha');


%%
% Excercise 3.1
trainFileID = fopen('IrisTrain2014.dt','r');
formatSpec = '%f %f %d';
sizeTrainMat = [3 Inf];
TrainMat = fscanf(trainFileID,formatSpec,sizeTrainMat);
TrainMat=TrainMat';

testFileID = fopen('IrisTest2014.dt','r');
formatSpec = '%f %f %d';
sizeTestMat = [3 Inf];
TestMat = fscanf(testFileID,formatSpec,sizeTestMat);
TestMat = TestMat';


% figure;

%display test input
for i = 1:size(TrainMat,1)
    flower = TrainMat(i,:);
    if(flower(3) == 0)
        hold on; plot(flower(:,1),flower(:,2),'b.');hold off;
    end
    
    if(flower(3) == 1)
        hold on; plot(flower(:,1),flower(:,2),'r.');hold off;
    end
    
    if(flower(3) == 2)
        hold on; plot(flower(:,1),flower(:,2),'y.');hold off;
    end
    
end



K = 5;

testRiskSum = 0;

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
        testRiskSum = testRiskSum + 1;
    end
    
end

testRiskSum = testRiskSum / size(TestMat,1);

testEfficacy = 1 - testRiskSum;

trainRiskSum = 0;

for i = 1:size(TrainMat,1)
    auxTrainMat = TrainMat;
    auxTrainMat(i,:) = [];
    distances = zeros(size(auxTrainMat,1),2);
    for j = 1:size(auxTrainMat,1)
        d = sqrt(power(auxTrainMat(j,1)-TrainMat(i,1),2)+power(auxTrainMat(j,2)-TrainMat(i,2),2));
        distances(j,:) = [d,auxTrainMat(j,3)]; 
    end
    distances = sortrows(distances,1);

    countingVector = zeros(1,3);
    for k = 1:K 
        countingVector(distances(k,2) + 1) = countingVector(distances(k,2) + 1) + 1;
    end
    
    index = find(countingVector == max(countingVector(:))) - 1;
  
    if index ~= TrainMat(i,3)
        trainRiskSum = trainRiskSum + 1;
    end
    
end

trainRiskSum = trainRiskSum / size(TrainMat,1);

trainEfficacy = 1 - trainRiskSum;


%%
clearvars;

trainFileID = fopen('IrisTrain2014.dt','r');
formatSpec = '%f %f %d';
sizeTrainMat = [3 Inf];
TrainMat = fscanf(trainFileID,formatSpec,sizeTrainMat);
TrainMat=TrainMat';

testFileID = fopen('IrisTest2014.dt','r');
formatSpec = '%f %f %d';
sizeTestMat = [3 Inf];
TestMat = fscanf(testFileID,formatSpec,sizeTestMat);
TestMat = TestMat';

S = 5;

kRiskMatrix = [];

blockDimension = int32(size(TrainMat,1)/S);

for k = 1:2:25
    
    kAverage = 0;
    
   for s = 1:S
        Test = TrainMat((s-1)*blockDimension + 1:s*blockDimension,:);
        Train = TrainMat;
        Train((s-1)*blockDimension + 1:s*blockDimension,:) = [];
        kAverage = kAverage + kNN(k,Train, Test); 
   end
   
   new_row =[k kAverage/5] ; 
   kRiskMatrix = [kRiskMatrix ; new_row];

end

    kBestIndex = find(kRiskMatrix == min(kRiskMatrix(2)));
    kBest = kRiskMatrix(kBestIndex,1);
    
    
    eNewRisk = kNN(kBest,TrainMat,TestMat);
    
    
    
    
%%
clearvars;

trainFileID = fopen('IrisTrain2014.dt','r');
formatSpec = '%f %f %d';
sizeTrainMat = [3 Inf];
TrainMat = fscanf(trainFileID,formatSpec,sizeTrainMat);
TrainMat=TrainMat';

testFileID = fopen('IrisTest2014.dt','r');
formatSpec = '%f %f %d';
sizeTestMat = [3 Inf];
TestMat = fscanf(testFileID,formatSpec,sizeTestMat);
TestMat = TestMat';


TrainMat2 = TrainMat(:,1:2);

%calcualte mean
trainS = sum(TrainMat2);
trainM = trainS/size(TrainMat2,1);

%calculate standard deviation
trainStdDev = [0 0];

for i = 1 : size(TrainMat2,1)
    x = TrainMat2(i,1);
    y = TrainMat2(i,2);
    trainStdDev = trainStdDev + [power(x-trainM(1),2) , power(y-trainM(2),2)];
end

trainStdDev = [sqrt(trainStdDev(1)/size(TrainMat2,1)), sqrt(trainStdDev(2)/size(TrainMat2,1))];

%normalize train set
TrainMatNormlaized = zeros(size(TrainMat2));

for i = 1 : size(TrainMat2,1)
    TrainMatNormalized(i,:) = (TrainMat2(i,:) - trainM)./trainStdDev;
end

%calculate mean and std dev for normalized values

%calcualte mean
trainSNorm = sum(TrainMatNormalized);
trainMNorm = trainSNorm/size(TrainMatNormalized,1);

%calculate standard deviation
trainNormStdDev = [0 0];

for i = 1 : size(TrainMatNormalized,1)
    x = TrainMatNormalized(i,1);
    y = TrainMatNormalized(i,2);
    trainNormStdDev = trainNormStdDev + [power(x-trainMNorm(1),2) , power(y-trainMNorm(2),2)];
end

trainNormStdDev = [sqrt(trainNormStdDev(1)/size(TrainMatNormalized,1)), sqrt(trainNormStdDev(2)/size(TrainMatNormalized,1))];


TrainMatNormalized = [TrainMatNormalized, TrainMat(:,3)];


%apply exercise 2.2.1 with normalized train data
S = 5;

kRiskMatrix = [];

blockDimension = int32(size(TrainMatNormalized,1)/S);

for k = 1:2:25
    
   kAverage = 0;
    
   for s = 1:S
        Test = TrainMatNormalized((s-1)*blockDimension + 1:s*blockDimension,:);
        Train = TrainMatNormalized;
        Train((s-1)*blockDimension + 1:s*blockDimension,:) = [];
        kAverage = kAverage + kNN(k,Train, Test); 
   end
   
   new_row =[k kAverage/5] ; 
   kRiskMatrix = [kRiskMatrix ; new_row];

end

%find kbest
kBestIndex = find(kRiskMatrix == min(kRiskMatrix(2)));
kBest = kRiskMatrix(kBestIndex,1);

% error on training data for k best
eTrainRisk = kNN(kBest,TrainMatNormalized,TrainMatNormalized);

%------------------------------

%normalize train set
TestMat2 = TestMat(:,1:2);

%calcualte mean
testS = sum(TestMat2);
testM = testS/size(TestMat2,1);

%calculate standard deviation
testStdDev = [0 0];

for i = 1 : size(TestMat2,1)
    x = TestMat2(i,1);
    y = TestMat2(i,2);
    testStdDev = testStdDev + [power(x-testM(1),2) , power(y-testM(2),2)];
end

testStdDev = [sqrt(testStdDev(1)/size(TestMat2,1)), sqrt(testStdDev(2)/size(TestMat2,1))];


%----------------

TestMatNormlaized = zeros(size(TestMat2));

for i = 1 : size(TestMat2,1)
    TestMatNormlaized(i,:) = (TestMat2(i,:) - trainM)./trainStdDev;
end

%Normalized meand and stdDev

%calcualte mean
testSNorm = sum(TestMatNormlaized);
testMNorm = testSNorm/size(TestMatNormlaized,1);

%calculate standard deviation
testNormStdDev = [0 0];

for i = 1 : size(TestMatNormlaized,1)
    x = TestMatNormlaized(i,1);
    y = TestMatNormlaized(i,2);
    testNormStdDev = testNormStdDev + [power(x-testMNorm(1),2) , power(y-testMNorm(2),2)];
end

testNormStdDev = [sqrt(testNormStdDev(1)/size(TestMatNormlaized,1)), sqrt(testNormStdDev(2)/size(TestMatNormlaized,1))];
        
TestMatNormlaized = [TestMatNormlaized, TestMat(:,3)];

eNewRisk = kNN(kBest,TrainMatNormalized,TestMatNormlaized);









