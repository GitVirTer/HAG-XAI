clear;

addpath(genpath(fullfile(pwd, 'util')));
Path.rawImage = 'Image_Classification_Stimuli_ImageNet';
Path.saveData = fullfile(pwd, 'saveTrainData_resnet50_Cls');
CheckIfDirExist(Path);
load('TrainingDatabase_GradAct_resnet50_Cls.mat', 'DataTrain', 'DataLabel');
% NFold = 5;
% cvIdx = crossvalind('KFold',numel(DataTrain),NFold);

load('cv5Idx.mat', 'cvIdx');
NFold = numel(unique(cvIdx));

h_max = 17; % 11 22 44
w_max = 13; % 19 38 76
c_max = 2048;


dirImage = dir(fullfile(Path.rawImage, ['*.jp*']));

%%
catNetCell = [];
for iFold = 1:NFold
    %%
    testIdx = cvIdx==iFold;
    XTest = DataTrain(testIdx);
    YTest = DataLabel(testIdx,:);
    dirTest = dirImage(testIdx);
    trainIdx = ~testIdx;
    XTrain = DataTrain(trainIdx);
    YTrain = DataLabel(trainIdx,:);

    %%
    layers = [
    sequenceInputLayer([h_max w_max c_max],"Name","input","Normalization","none","MinLength", 2)
%     learnableReLuLayer('Name', 'learnableReLuLayer')
%     learnableGaussianConvReLuNoNormLayer(Name="LXAI", p_phi=true, p_smooth=true, p_norm=true, rawSize=[576 1024])
    learnableGaussianConvReLuNoNormLayer()

%     PCCRegressionLayer("regressionoutput")
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

    nameNetCell = struct2cell(dirTest)';
    nameNetCell = nameNetCell(:,1);
    netXAI = net;
    nameNetCell(:,2) = {netXAI};
    catNetCell = cat(1, catNetCell, nameNetCell);

    save(fullfile(Path.saveData, ['Results_Fold-' num2str(iFold) '.mat']), "Results", 'net','trainInfo','opt','NFold');

end
save(fullfile(Path.saveData, ['Results_allNet.mat']), "Results", 'catNetCell','opt','NFold');

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


