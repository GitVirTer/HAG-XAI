classdef learnableGaussianConvReLuNoNormLayer_correct < nnet.layer.Layer & nnet.layer.Formattable
    % Example custom PReLU layer.

    properties
        % (Optional) Layer properties.
    
        % Declare layer properties here.
        rawSize
        gsize

        R
        C
        p_phi
        p_smooth
        p_norm
        
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

        ObjW
        G_A_grad
        G_sigma_grad
        G_A_all
        G_sigma_all

    end

    methods
        function layer = learnableGaussianConvReLuNoNormLayer_correct(args)
            % layer = preluLayer creates a learnableReLuLayer layer.
            %
            % layer = preluLayer(numChannels,Name=name) also specifies the
            % layer name.

            arguments
                args.Name = "LXAILayer";
                args.p_phi = true;
                args.p_smooth = true;
                args.p_norm = true;
                args.rawSize = [576 1024];
            end

            % Set layer name.
            layer.Name = args.Name;
%             layer.p_phi = args.p_phi;
%             layer.p_smooth = args.p_smooth;
%             layer.p_norm = args.p_norm;

            % Set layer description.
            layer.Description = "learnableConvReLuLayer";
%             layer.rawSize = args.rawSize;   %[576 1024];
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
%             layer.Alpha_grad_neg = ones(szAlpha);       

            % Initialize Conv Weights
            layer.gsize = [21 21];
%             layer.convW_act = ones(filterSize);
%             layer.convW_grad = ones(filterSize);
            [layer.R, layer.C] = ndgrid(1:layer.gsize(1), 1:layer.gsize(2));
            layer.G_A_grad = 1;
            layer.G_sigma_grad = 3;
            layer.G_A_all = 1;
            layer.G_sigma_all = 3;            
%             gmat = layer.G_A .* exp(-((layer.R-round(layer.gsize(1)/2)).^2 + (layer.C-round(layer.gsize(2)/2)).^2)./(2*abs(layer.G_sigma) + 1e-5));
            
            layer.ObjW = 1;


            layer.p_phi = true;
            layer.p_smooth = true;
            layer.p_norm = true;
            layer.rawSize = [576 1024];

        end

        function Z = predict(layer, X)
            % Z = predict(layer, X) forwards the input data X through the
            % layer and outputs the result Z.
            eps_val = 1e-5;
            % X:SSCBT
            Act = X(:,:,:,:,1:2:end);
            Grads = X(:,:,:,:,2:2:end);


            if layer.p_phi
                Act_oper = layer.Alpha_act_pos .* max(0, Act) + layer.Alpha_act_neg .* min(0, Act); % SSCBT, T=1
                Grads_oper = layer.Alpha_grad_pos .* max(0, Grads) + layer.Alpha_grad_neg .* min(0, Grads); % SSCBT
            else
                Act_oper = 1 .* max(0, Act) + 1 .* min(0, Act); % SSCBT, T=1
                Grads_oper = 1 .* max(0, Grads) + 0 .* min(0, Grads); % SSCBT
%                 Act_oper = Act;
%                 Grads_oper = Grads;      
            end

%             tmp_convW_actx = layer.G_A .* exp(-((layer.R-round(layer.gsize(1)/2)).^2 + (layer.C-round(layer.gsize(2)/2)).^2)./(2*abs(layer.G_sigma) + 1e-5));
%             convW_actx = repmat(tmp_convW_actx, [1 1 1 1 size(Act_oper,finddim(Act_oper,"C"))]); % SSC
%             Act_oper = dlconv(Act_oper, convW_actx, 0,"Padding","same");

            if layer.p_smooth
                tmp_convW_gradx = layer.G_A_grad .* exp(-((layer.R-round(layer.gsize(1)/2)).^2 + (layer.C-round(layer.gsize(2)/2)).^2)./(2*abs(layer.G_sigma_grad) + 1e-5));
                convW_gradx = repmat(tmp_convW_gradx, [1 1 1 1 size(Grads_oper,finddim(Grads_oper,"C"))]); % SSC
                Grads_oper = dlconv(Grads_oper, convW_gradx, 0,"Padding","same");
            end

            Act_Grad = Act_oper.*Grads_oper;    % SSCBT
            Act_Grad = sum(Act_Grad, finddim(Act_Grad,"C"));
            Act_Grad = max(0, Act_Grad);    % ReLu
%             saliencyMap = dlresize(saliencyMap,'OutputSize',layer.rawSize,'Method','linear');

            % Normalization
%             saliencyMap_ObjW = sum(Act_Grad, [1 2 3]);
            if layer.p_norm
                saliencyMap_ObjW = sum(Act_Grad, [1 2 3]).*layer.ObjW;
                Act_Grad = Act_Grad.*(1./(saliencyMap_ObjW+1e-5));
            else
                Act_Grad = (Act_Grad-min(Act_Grad,[],finddim(Act_Grad,"S")))./...
                    (max(Act_Grad,[],finddim(Act_Grad,"S"))-min(Act_Grad,[],finddim(Act_Grad,"S"))+eps_val);
            end

            saliencyMap = sum(Act_Grad, finddim(Act_Grad,"T"));

            % Smoothing
            if layer.p_smooth
                tmp_convW_all = layer.G_A_all .* exp(-((layer.R-round(layer.gsize(1)/2)).^2 + (layer.C-round(layer.gsize(2)/2)).^2)./(2*abs(layer.G_sigma_all) + 1e-5));
                convW_all = repmat(tmp_convW_all, [1 1 1 1 size(saliencyMap,finddim(saliencyMap,"C"))]); % SSC
                saliencyMap = dlconv(saliencyMap, convW_all, 0,"Padding","same");
            end

            saliencyMap = (saliencyMap-min(saliencyMap,[],finddim(saliencyMap,"S")))./...
                (max(saliencyMap,[],finddim(saliencyMap,"S"))-min(saliencyMap,[],finddim(saliencyMap,"S"))+eps_val);            
            saliencyMap = dlarray(saliencyMap, 'SSCB');
            saliencyMap = dlresize(saliencyMap,'OutputSize',layer.rawSize,'Method','linear');
           

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




