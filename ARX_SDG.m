%% Synthetic Data Generation (SDG) by nonlinear AutoRegressive with eXogenous input (ARX) model
% Developed by Seyed Muhammad Hossein Mousavi - July 2023
% ARX models could be used for prediction and forecasting future. As, data generated for the future 
% is similar to past/original data, it could be used for synthetic data generation.
% An ARX model, which stands for AutoRegressive with eXogenous input model, is a type of linear
% time-series model commonly used in statistics and econometrics for modeling and forecasting data.
% It falls under the broader category of autoregressive models.
% A nonlinear ARX (AutoRegressive with eXogenous input) model is a type of time series model that 
% extends the traditional linear ARX model to accommodate non-linear relationships between the
% variables. In a nonlinear ARX model, the relationship between the current value of the time series 
% and its lagged values, as well as the exogenous variables, is expressed in a nonlinear form.
% ---------------------------------------------------------------------------------------
clear;
close all;
% Loading data
load fisheriris.mat;
data=reshape(meas,1,[]); % Preprocessing - convert matrix to vector
data=data'; Input=meas;
SS = size(Input); SF = SS (1,2); SS = SS (1,1); % Number of samples and features
Target(1:50)=1;Target(51:100)=2;Target(101:150)=3;Target=Target'; % Original labels
% “G_Amount” is the amount of generated samples. 1 is equal to the number of original samples.
% 2 is double the original samples and goes on with this sequence. 
G_Amount = 5;
Change = 4; % Variation level in the generated data
AddVar = 4; % Additive variation in each iteration
Orders = 128; %The number of delayed outputs
% Loop for generating synthetic dataset
for i = 1:G_Amount
sysNL = nlarx(data,Orders);
ToGen{i} = compare(data,sysNL,Change);
Change = Change + AddVar;
disp([' Generating Pack Number "',num2str(i)]);
end
% Converting cell to matrix
for i = 1 : G_Amount
Generated(:,i)=ToGen{i};
end
% Converting matrix to cell
for i = 1 : G_Amount
Generated1{i}=reshape(Generated(:,i),[SS,SF]);
Generated1{i}(:,end+1)=Target; % Adding labels
end
% Converting cell to matrix (the last time)
Synthetic = cell2mat(Generated1');
SyntheticData= Synthetic (:,1:end-1); % Synthetic dataset features
SyntheticLabel= Synthetic (:,end); % Synthetic dataset labels

%% Plot data and classes
Feature1=3;
Feature2=4;
f1=Input(:,Feature1); % feature1
f2=Input(:,Feature2); % feature 2
ff1=SyntheticData(:,Feature1); % feature1
ff2=SyntheticData(:,Feature2); % feature 2
figure('units','normalized','outerposition',[0 0 1 1])
subplot(2,2,1)
plot(Input, 'linewidth',1); title('Original Data');
ax = gca; ax.FontSize = 12; ax.FontWeight='bold'; grid on;
subplot(2,2,2)
plot(SyntheticData, 'linewidth',1); title('Synthetic Data');
ax = gca; ax.FontSize = 12; ax.FontWeight='bold'; grid on;
subplot(2,2,3)
gscatter(f1,f2,Target,'rkgb','.',20); title('Original');
ax = gca; ax.FontSize = 12; ax.FontWeight='bold'; grid on;
subplot(2,2,4)
gscatter(ff1,ff2,SyntheticLabel,'rkgb','.',20); title('Synthetic');
ax = gca; ax.FontSize = 12; ax.FontWeight='bold'; grid on;

%% Train and Test
% Training Synthetic dataset by SVM
Mdlsvm  = fitcecoc(SyntheticData,SyntheticLabel); CVMdlsvm = crossval(Mdlsvm); 
SVMError = kfoldLoss(CVMdlsvm); SVMAccAugTrain = (1 - SVMError)*100;
% Predict new samples on whole original dataset by SVM
[label5,score5,cost5] = predict(Mdlsvm,Input);
sizlbl=size(Target); sizlbl=sizlbl(1,1);
countersvm=0; % Misclassifications places
misindexsvm=0; % Misclassifications indexes
for i=1:sizlbl
if Target(i)~=label5(i)
misindex(i)=i; countersvm=countersvm+1; end; end
% Testing the accuracy
TestErrAugsvm = countersvm*100/sizlbl; SVMAccAugTest = 100 - TestErrAugsvm;
% Result SVM
AugResSVM = [' Synthetic Train SVM "',num2str(SVMAccAugTrain),'" Synthetic Test SVM"', num2str(SVMAccAugTest),'"'];
disp(AugResSVM);
