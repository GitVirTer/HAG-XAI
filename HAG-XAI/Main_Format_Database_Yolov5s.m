clear;

dataPath = '..\';   % The root path of this tool
Path.humanSaliencyMap = fullfile(dataPath, 'Data', 'human_attention_data');
Path.aiRawData = fullfile(dataPath, 'Yolov5s\Yolov5s, FullGradCAM++, Layer1\', 'saveRawGradAct_vehicle_NMS_class_F1_nofaith_norm_yolov5sbdd100k300epoch_1');

dir_ai = dir(fullfile(Path.aiRawData, '*.mat'));

validImageInfo = {};
DataTrain = {};
DataLabel = [];
cnt = 0;
for i = 1:numel(dir_ai)
    curFileName = dir_ai(i).name;
    curFileNum = split(curFileName, '-');
    curFileNum = curFileNum{1};
    mat_load_ai = load(fullfile(Path.aiRawData, curFileName));
    mat_load_human = load(fullfile(Path.humanSaliencyMap, [curFileNum '_GSmo_30.mat']));
    curLabel = mat_load_human.output_map_norm(:);   % 576*1024

    h_max = 44; % 11 22 44
    w_max = 76; % 19 38 76
    c_max = 512;
    if isempty(mat_load_ai.grad_act)
        continue;
    end

    validImageInfo(i,:) = {curFileNum, fullfile(Path.humanSaliencyMap, [curFileNum '_GSmo_30.mat']), fullfile(Path.aiRawData, curFileName)};
    grad_act_format = [];
    for iObj = 1:numel(mat_load_ai.grad_act)
        grad = permute(squeeze(mat_load_ai.grad_act{iObj}(1,1,:,:,:)), [2 3 1]); % H W C
        act = permute(squeeze(mat_load_ai.grad_act{iObj}(2,1,:,:,:)), [2 3 1]);  % H W C
%         if min(act, [], 'all') < 0
%             warning('Act has negative value');
%         end
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
    % grad_act_format = cat(4, grad_act_format, act); % Grad, Grad...Act
    % grad_act_format = flip(grad_act_format,4);      % Act, Grad, Grad...

    cnt = cnt+1;
    DataTrain{cnt,1} = grad_act_format;
    DataLabel(cnt,:) = curLabel;

    disp(num2str(i));

end

save TrainingDatabase_HAGXAI_Yolov5s.mat DataTrain DataLabel validImageInfo -v7.3

