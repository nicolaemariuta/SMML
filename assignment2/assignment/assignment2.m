clearvars;
%read data
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

%normalize train data setdata set
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

%normalize 
TrainNorm = zeros(size(TrainMat2));

for i = 1 : size(TrainMat2,1)
    TrainNorm(i,:) = (TrainMat2(i,:) - m)./stdDev;
end

TrainMatNormalized = [TrainNorm, TrainMat(:,3)];

%normalize test set

    TestMat2 = TestMat(:,1:2);
    TestMatNormlaized = zeros(size(TestMat2));

    for i = 1 : size(TestMat2,1)
        TestMatNormlaized(i,:) = (TestMat2(i,:) - m)./stdDev;
    end
    
    TestMatNormlaized = [TestMatNormlaized, TestMat(:,3)];




%calculate mean for the 3 classes of flowers
sum0 = zeros(1,2);
sum1 = zeros(1,2);
sum2 = zeros(1,2);
count0 = 0;
count1 = 0;
count2 = 0;

for i = 1:size(TrainMatNormalized,1);
    if TrainMatNormalized(i,3) == 0
        sum0 = sum0 + [TrainMatNormalized(i,1),TrainMatNormalized(i,2)];
        count0 = count0+1;
    elseif TrainMatNormalized(i,3) == 1
        sum1 = sum1 + [TrainMatNormalized(i,1),TrainMatNormalized(i,2)];
        count1 = count1+1;
    elseif TrainMatNormalized(i,3) == 2
        sum2 = sum2 + [TrainMatNormalized(i,1),TrainMatNormalized(i,2)];
        count2 = count2+1;
    end
end

mean0 = sum0/count0;
mean1 = sum1/count1;
mean2 = sum2/count2;

%computing scatter matrices

%within-class scatter matrix
scatter0 = zeros(1,2);
scatter1 = zeros(1,2);
scatter2 = zeros(1,2);

for i = 1:size(TrainMatNormalized,1);
    if TrainMatNormalized(i,3) == 0
        scatter0 = scatter0 + (TrainNorm(i,:))*(TrainNorm(i,:)-mean0)';       
    elseif TrainMatNormalized(i,3) == 1
        scatter1 = scatter1 + (TrainNorm(i,:)-mean1)*(TrainNorm(i,:)-mean1)';   
    elseif TrainMatNormalized(i,3) == 2
        scatter2 = scatter2 + (TrainNorm(i,:)-mean2)*(TrainNorm(i,:)-mean2)';   
    end
end

withinClassScatterMatrix = [scatter0;scatter1;scatter2];


%between-class scatter matrix
bscatter0 = count0 +(mean0 - withinClassScatterMatrix(1,:))*(mean0 - withinClassScatterMatrix(1,:))'+(mean1 - withinClassScatterMatrix(1,:))*(mean1 - withinClassScatterMatrix(1,:))'+(mean2 - withinClassScatterMatrix(1,:))*(mean2 - withinClassScatterMatrix(1,:))';
bscatter1 = count1 +(mean0 - withinClassScatterMatrix(2,:))*(mean0 - withinClassScatterMatrix(2,:))'+(mean1 - withinClassScatterMatrix(2,:))*(mean2 - withinClassScatterMatrix(2,:))'+(mean2 - withinClassScatterMatrix(2,:))*(mean2 - withinClassScatterMatrix(2,:))';
bscatter2 = count2 +(mean0 - withinClassScatterMatrix(3,:))*(mean0 - withinClassScatterMatrix(3,:))'+(mean1 - withinClassScatterMatrix(3,:))*(mean3 - withinClassScatterMatrix(3,:))'+(mean2 - withinClassScatterMatrix(3,:))*(mean2 - withinClassScatterMatrix(3,:))';

betweenClassScatterMatrix = [bscatter0;bscatter1;bscatter2];







