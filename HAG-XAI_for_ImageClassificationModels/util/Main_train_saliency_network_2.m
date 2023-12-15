clear;
addpath(genpath(pwd));


load ai_human_database_v5l_0_1_12_layer_1.mat human_data ai_data;

NFold = 5;
% cvIdx = crossvalind('KFold',size(human_data, 4),NFold);
load('tmpNetResults_1.mat', 'cvIdx');

%%
for iFold = 1:NFold
    %%
    testIdx = cvIdx==iFold;
    XTest = ai_data(:,:,:,testIdx);
    YTest = human_data(:,:,:,testIdx);
    trainIdx = ~testIdx;
    XTrain = ai_data(:,:,:,trainIdx);
    YTrain = human_data(:,:,:,trainIdx);

    %%
    layers = [
    imageInputLayer([576 1024 9],"Name","imageinput","Normalization","none")
    groupedConvolution2dLayer([5 5],1,"channel-wise","Name","groupedconv","BiasLearnRateFactor",0,"Padding","same")
%     convolution2dLayer([5 5],16,"Name","groupedconv","BiasLearnRateFactor",0,"Padding","same")
    convolution2dLayer([1 1],1,"Name","conv","BiasLearnRateFactor",0,"Padding","same","WeightsInitializer","ones")
    reluLayer("Name","reluLayer_1")
%     sigmoidLayer("Name","reluLayer_1")
%     PCCRegressionLayer("regressionoutput")
%     mseRegressionLayer("regressionoutput")
    PCCMSERegressionLayer("regressionoutput")
    ];
    lgraph = layerGraph(layers);

    %%
    opt.XTest = XTest;
    opt.YTest = YTest;
    [net, trainInfo] = trainModel_Custom(XTrain,YTrain,lgraph, opt);

    %%
    Results{iFold}.net = net;
    Results{iFold}.trainInfo = trainInfo;

end

