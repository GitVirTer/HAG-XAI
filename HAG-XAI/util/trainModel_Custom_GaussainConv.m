function [net, trainInfo] = trainModel_Custom_GaussainConv(XTrain,YTrain,lgraph, opt)

% funcHandle = @waveletConvLayer_varLen;
% lgraph = net_CNN_WaveletConvTest_2(funcHandle);

% plotNetWeight(lgraph);
% drawnow;

maxEpochs = 120;
miniBatchSize = 40;
validationFrequency = 3*ceil(size(YTrain,1)/miniBatchSize);
% validationFrequency = 10;

lr_start = 5e-2;
lr_end = 1e-2;
lr_drop_period = 20;
lr_factor = (lr_end/lr_start)^(1/(maxEpochs/lr_drop_period));

checkpointPath = 'H:\work\Iris\checkPoint_1\';
options = trainingOptions('adam', ...
'L2Regularization', 0, ...
'InitialLearnRate',lr_start, ...
'MaxEpochs',maxEpochs, ...
'MiniBatchSize',miniBatchSize, ...
'LearnRateSchedule','piecewise',...
'LearnRateDropFactor',lr_factor,...
'LearnRateDropPeriod',lr_drop_period,...
'ValidationData', {opt.XTest, opt.YTest},...
'ValidationFrequency',validationFrequency, ...
'Shuffle','every-epoch', ...
'ValidationPatience',10,...
'Verbose',true, ...
'ExecutionEnvironment', 'gpu',...
'Plots','none',...
'CheckpointPath', []);    %checkpointPath
% 'ValidationData', {XTest, YTest},...
% 'Plots','training-progress',...
% 'GradientThreshold', 10, ...

[net, trainInfo] = trainNetwork(XTrain,YTrain,lgraph,options);
% plotNetWeight(net);

end

