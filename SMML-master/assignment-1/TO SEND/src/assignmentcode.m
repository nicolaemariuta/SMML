
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

mu = [1 2];
sigma = [0.3 0.2; 0.2 0.2];
R = chol(sigma,'lower');
z = repmat(mu',1,100) + R*randn(100,2)';
z=z';

plot(z(:,1),z(:,2),'b.');

%%
%Exercise 2.3
%calcualte mean

mu = [1 2];
sigma = [0.3 0.2; 0.2 0.2];
R = chol(sigma,'lower');
z = repmat(mu',1,100) + R*randn(100,2)';
z=z';

plot(z(:,1),z(:,2),'b.');

s = sum(z);
m = s/size(z,1);

hold on; plot(m(:,1),m(:,2),'r.');hold off;
hold on; plot(mu(:,1),mu(:,2),'g.');hold off;

%%
%Exercise 2.4
%calculate covariance
mu = [1 2];
sigma = [0.3 0.2; 0.2 0.2];
R = chol(sigma,'lower');
z = repmat(mu',1,100) + R*randn(100,2)';
z=z';

plot(z(:,1),z(:,2),'b.');

s = sum(z);
m = s/size(z,1);

hold on; plot(m(:,1),m(:,2),'r.');hold off;
hold on; plot(mu(:,1),mu(:,2),'g.');hold off;

covariance = cov(z);
%covariance=sigma;
[eigenvec, eigenval] = eig(covariance);
k = sqrt(diag(eigenval));
axis([-1 4 -1 4])
hold on; plot([m(1) m(1)+k(1)*eigenvec(1,1)],[m(2) m(2)+k(1)*eigenvec(1,2)],'r-');hold off;
hold on; plot([m(1) m(1)+k(2)*eigenvec(2,1)],[m(2) m(2)+k(2)*eigenvec(2,2)],'r-');hold off;


%%
%Rotated Gaussian distributions
figure;
mu = [1 2];
sigma = [0.3 0.2; 0.2 0.2];

rad30 = 30*pi/180;
rad60 = 60*pi/180;
rad90 = 90*pi/180;

r30 = [cos(rad30) -sin(rad30); sin(rad30) cos(rad30)];
r60 = [cos(rad60) -sin(rad60); sin(rad60) cos(rad60)];
r90 = [cos(rad90) -sin(rad90); sin(rad90) cos(rad90)];
sigma30 = r30 * sigma *inv(r30) ;
sigma60 = r60 * sigma *inv(r60) ;
sigma90 = r90 * sigma *inv(r90) ;

R30 = chol(sigma30,'lower');
R60 = chol(sigma60,'lower');
R90 = chol(sigma90,'lower');

z30 = repmat(mu',1,100) + R30*randn(100,2)';
z60 = repmat(mu',1,100) + R60*randn(100,2)';
z90 = repmat(mu',1,100) + R90*randn(100,2)';

z30=z30';
z60=z60';
z90=z90';


plot(z30(:,1),z30(:,2),'b.');
hold on;plot(z60(:,1),z60(:,2),'y.');hold off;
hold on;plot(z90(:,1),z90(:,2),'r.');hold off;


%Exercise 2.4
%calculate covariance
covariance30 = cov(z30);
covariance60 = cov(z60);
covariance90 = cov(z90);
%covariance=sigma;
[eigenvec30, eigenval30] = eig(covariance30);
[eigenvec60, eigenval60] = eig(covariance60);
[eigenvec90, eigenval90] = eig(covariance90);

k30 = sqrt(diag(eigenval30));
k60 = sqrt(diag(eigenval60));
k90 = sqrt(diag(eigenval90));

axis([-1 4 -1 4])
hold on; plot([m(1) m(1)+k30(1)*eigenvec30(1,1)],[m(2) m(2)+k30(1)*eigenvec30(1,2)],'b-');hold off;
hold on; plot([m(1) m(1)+k30(2)*eigenvec30(2,1)],[m(2) m(2)+k30(2)*eigenvec30(2,2)],'b-');hold off;

axis([-1 4 -1 4])
hold on; plot([m(1) m(1)+k60(1)*eigenvec60(1,1)],[m(2) m(2)+k60(1)*eigenvec60(1,2)],'y-');hold off;
hold on; plot([m(1) m(1)+k60(2)*eigenvec60(2,1)],[m(2) m(2)+k60(2)*eigenvec60(2,2)],'y-');hold off;

axis([-1 4 -1 4])
hold on; plot([m(1) m(1)+k90(1)*eigenvec90(1,1)],[m(2) m(2)+k90(1)*eigenvec90(1,2)],'r-');hold off;
hold on; plot([m(1) m(1)+k90(2)*eigenvec90(2,1)],[m(2) m(2)+k90(2)*eigenvec90(2,2)],'r-');hold off;


%%
%Excercise 3.1

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


figure;

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

riskSum = 0;

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
        riskSum = riskSum + 1;
    end
    
end

riskSum = riskSum / size(TestMat,1);

%%
%Exercise 3.2
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

[minVal minInd] = min(kRiskMatrix);
kBest = kRiskMatrix(minInd(2),1);
    
eNewRisk = kNN(kBest,TrainMatNormalized,TestMatNormlaized);

%%
%Excerise 3.3
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
s = sum(TrainMat2);
m = s/size(TrainMat2,1);

%calculate standard deviation
stdDev = [0 0];

for i = 1 : size(TrainMat2,1)
    x = TrainMat2(i,1);
    y = TrainMat2(i,2);
    stdDev = stdDev + [power(x-m(1),2) , power(y-m(2),2)];
end

stdDev = [sqrt(stdDev(1)/size(TrainMat2,1)), sqrt(stdDev(2)/size(TrainMat2,1))];

%normalize train set
TrainMatNormlaized = zeros(size(TrainMat2));

for i = 1 : size(TrainMat2,1)
    TrainMatNormalized(i,:) = (TrainMat2(i,:) - m)./stdDev;
end

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

[minVal minInd] = min(kRiskMatrix);
kBest = kRiskMatrix(minInd(2),1);

%normalize train set
TestMat2 = TestMat(:,1:2);
TestMatNormlaized = zeros(size(TestMat2));

for i = 1 : size(TestMat2,1)
    TestMatNormlaized(i,:) = (TestMat2(i,:) - m)./stdDev;
end

TestMatNormlaized = [TestMatNormlaized, TestMat(:,3)];

eNewRisk = kNN(kBest,TrainMatNormalized,TestMatNormlaized);


