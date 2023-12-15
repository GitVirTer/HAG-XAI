clear;
addpath(genpath(pwd));

% Load Data
mat_load_ai = load('11-res.jpg.mat');

% Settings for Yolov5s
h_max = 44; % 11 22 44
w_max = 76; % 19 38 76
c_max = 512;
if isempty(mat_load_ai.grad_act)
    error('No Detected Objects');
end
RawSize = [576 1024];

% Format Testing Data
grad_act_format = [];
for iObj = 1:numel(mat_load_ai.grad_act)
    grad = permute(squeeze(mat_load_ai.grad_act{iObj}(1,1,:,:,:)), [2 3 1]); % H W C
    act = permute(squeeze(mat_load_ai.grad_act{iObj}(2,1,:,:,:)), [2 3 1]);  % H W C
    h = size(grad,1);
    w = size(grad,2);
    c = size(grad,3);
    if ~isequal([h w], [h_max w_max])
        warning('Resized Feature Map');
        grad = extractdata(dlresize(dlarray(grad, 'SSC'),'OutputSize',[h_max w_max]));
        act = extractdata(dlresize(dlarray(act, 'SSC'),'OutputSize',[h_max w_max]));
    end
    if ~isequal(c, c_max)
        warning('Add Empty Channels Feature Map');
        padMat = zeros([size(grad,1) size(grad,2) c_max-c]);
        grad = cat(3, grad, padMat);
        act = cat(3, act, padMat);
    end
    grad_act_format = cat(4, grad_act_format, act, grad);
end

% Inference using HAG-XAI
load('Results_Fold-1.mat', 'net');
masks_ndarray = predict(net, {grad_act_format}, 'ExecutionEnvironment','cpu','MiniBatchSize',30);
masks_ndarray = reshape(masks_ndarray, RawSize);
figure;
imagesc(masks_ndarray);

% Save Data
clearvars -except masks_ndarray
load('11-res.jpg.mat');
save('11-HAGXAI.mat')


