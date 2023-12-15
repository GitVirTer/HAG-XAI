classdef learnableConvWeightedReLuLayer < nnet.layer.Layer & nnet.layer.Formattable
    % Example custom PReLU layer.

    properties
        % (Optional) Layer properties.
    
        % Declare layer properties here.
        rawSize

    end

    properties (Learnable)
        % Layer learnable parameters.

        % Scaling coefficient.
        Alpha_act_pos
        Alpha_act_neg
        Alpha_grad_pos
        Alpha_grad_neg     
        convW_act
        convW_grad

    end

    methods
        function layer = learnableConvWeightedReLuLayer(args)
            % layer = preluLayer creates a learnableReLuLayer layer.
            %
            % layer = preluLayer(numChannels,Name=name) also specifies the
            % layer name.

            arguments
                args.Name = "";
            end

            % Set layer name.
            layer.Name = args.Name;

            % Set layer description.
            layer.Description = "learnableConvReLuLayer";
            layer.rawSize = [576 1024];
        end

        function layer = initialize(layer,layout)
            % layer = initialize(layer,layout) initializes the learnable
            % parameters of the layer for the specified input layout.

            % Skip initialization of nonempty parameters.
            if (~isempty(layer.Alpha_act_pos)) && (~isempty(layer.Alpha_act_neg))
                return
            end

            % Input data size.
            sz = layout.Size;
            ndims = numel(sz);

            % Find number of channels.
            idx = finddim(layout,"C");
            numChannels = sz(idx);

            % Initialize Alpha.
            szAlpha = ones(1,ndims);
            szAlpha(idx) = 1; % =1: global ReLu. =numChannels: channel-wise ReLu
            layer.Alpha_act_pos = ones(szAlpha);
            layer.Alpha_act_neg = ones(szAlpha);
            layer.Alpha_grad_pos = ones(szAlpha);
            layer.Alpha_grad_neg = zeros(szAlpha);       

            % Initialize Conv Weights
            filterSize = [11 11];
            layer.convW_act = ones(filterSize);
            layer.convW_grad = ones(filterSize);

        end

        function Z = predict(layer, X)
            % Z = predict(layer, X) forwards the input data X through the
            % layer and outputs the result Z.
            eps_val = 1e-5;
            % X:SSCBT
            Act = X(:,:,:,:,1);
            Grads = X(:,:,:,:,2:end);

            Act_oper = layer.Alpha_act_pos .* max(0, Act) + layer.Alpha_act_neg .* min(0, Act); % SSCBT, T=1
            Grads_oper = layer.Alpha_grad_pos .* max(0, Grads) + layer.Alpha_grad_neg .* min(0, Grads); % SSCBT

            convW_actx = repmat(layer.convW_act, [1 1 1 1 size(Act_oper,finddim(Act_oper,"C"))]); % SSC
            Act_oper = dlconv(Act_oper, convW_actx, 0,"Padding","same");

            convW_gradx = repmat(layer.convW_grad, [1 1 1 1 size(Grads_oper,finddim(Grads_oper,"C"))]); % SSC
            Grads_oper = dlconv(Grads_oper, convW_gradx, 0,"Padding","same");

            Act_oper = Act_oper.*maxpool(Grads_oper,'global');

            Act_Grad = Act_oper.*Grads_oper;    % SSCBT
            Act_Grad = sum(Act_Grad, finddim(Act_Grad,"C"));
            Act_Grad = max(0, Act_Grad);    % ReLu

            % Normalization
            Act_Grad = (Act_Grad-min(Act_Grad,[],finddim(Act_Grad,"S")))./...
                (max(Act_Grad,[],finddim(Act_Grad,"S"))-min(Act_Grad,[],finddim(Act_Grad,"S"))+eps_val);
            saliencyMap = sum(Act_Grad, finddim(Act_Grad,"T"));
            saliencyMap = (saliencyMap-min(saliencyMap,[],finddim(saliencyMap,"S")))./...
                (max(saliencyMap,[],finddim(saliencyMap,"S"))-min(saliencyMap,[],finddim(saliencyMap,"S"))+eps_val);            
            saliencyMap = dlarray(saliencyMap, 'SSCB');
            saliencyMap = dlresize(saliencyMap,'OutputSize',layer.rawSize);
            

            % Flatten
            saliencyMap = stripdims(saliencyMap);
            saliencyMap = permute(saliencyMap, [4 1 2 3]);
            saliencyMap = saliencyMap(:,:); % BC
            saliencyMap = saliencyMap';     % CB
            saliencyMap = dlarray(saliencyMap, 'CB');

            Z = saliencyMap;

        end
    end
end




