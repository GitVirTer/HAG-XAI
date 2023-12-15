clear;

addpath(genpath(fullfile(pwd, 'util')));

Path.saveData = fullfile(pwd, 'saveData_5Fold_HAGXAI');
CheckIfDirExist(Path);
load('TrainingDatabase_HAGXAI_Yolov5s.mat', 'DataTrain', 'DataLabel');

NFold = 5;
cvIdx = crossvalind('KFold',numel(DataTrain),NFold);

% load('cv5Idx.mat', 'cvIdx');
% NFold = numel(unique(cvIdx));

h_max = 44; % 11 22 44
w_max = 76; % 19 38 76
c_max = 512;

%%
for iFold = 1:NFold
    %%
    testIdx = cvIdx==iFold;
    XTest = DataTrain(testIdx);
    YTest = DataLabel(testIdx,:);
    trainIdx = ~testIdx;
    XTrain = DataTrain(trainIdx);
    YTrain = DataLabel(trainIdx,:);

    %%
    layers = [
    sequenceInputLayer([h_max w_max c_max],"Name","input","Normalization","none","MinLength", 2)
    learnableGaussianConvReLuNoNormLayer_correct('Name', 'learnableReLuLayer')
    PCCMSERegressionLayer("regressionoutput")
    ];
    lgraph = layerGraph(layers);

    %%
    opt.XTest = XTest;
    opt.YTest = YTest;
    [net, trainInfo] = trainModel_Custom_GaussainConv(XTrain,YTrain,lgraph, opt);

    %%
    Results{iFold}.net = net;
    Results{iFold}.trainInfo = trainInfo;

    save(fullfile(Path.saveData, ['Results_Fold-' num2str(iFold) '.mat']), "Results", 'net','trainInfo','opt','NFold');

end

% save TrainingDatabase_F1_GradAct_ResultsCV10.mat Results cvIdx

%%
figure;
for iNet = 1:numel(Results)
    subplot(2, numel(Results), iNet);
    imagesc(Results{1, iNet}.net.Layers(2, 1).convW_act); colorbar();
    subplot(2, numel(Results), numel(Results) + iNet);
    imagesc(Results{1, iNet}.net.Layers(2, 1).convW_grad); colorbar();
    Alpha_act_pos(iNet,:) = squeeze(Results{1, iNet}.net.Layers(2, 1).Alpha_act_pos);
    Alpha_act_net(iNet,:) = squeeze(Results{1, iNet}.net.Layers(2, 1).Alpha_act_neg);
    Alpha_grad_pos(iNet,:) = squeeze(Results{1, iNet}.net.Layers(2, 1).Alpha_grad_pos);
    Alpha_grad_net(iNet,:) = squeeze(Results{1, iNet}.net.Layers(2, 1).Alpha_grad_neg);

end

figure;
% bar([Alpha_act_pos Alpha_grad_pos Alpha_grad_net]);

for iNet = 1:numel(Results)
    subplot(1, numel(Results), iNet);
    histogram(Alpha_grad_net(iNet,:));   %Alpha_act_pos Alpha_act_net Alpha_grad_pos Alpha_grad_net

end
