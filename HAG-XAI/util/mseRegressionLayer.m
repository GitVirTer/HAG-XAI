classdef mseRegressionLayer < nnet.layer.RegressionLayer ...
        & nnet.layer.Acceleratable
    % Example custom regression layer with mean-absolute-error loss.
    
    methods
        function layer = mseRegressionLayer(name)
            % layer = maeRegressionLayer(name) creates a
            % mean-absolute-error regression layer and specifies the layer
            % name.
			
            % Set layer name.
            layer.Name = name;

            % Set layer description.
            layer.Description = 'Mean absolute error';
        end
        
        function loss = forwardLoss(layer, Y, T)
            % loss = forwardLoss(layer, Y, T) returns the MAE loss between
            % the predictions Y and the training targets T.

            % Flatten


            % Calculate MAE.
%             R = size(Y,3);
            N = size(Y,4);

            Y = reshape(Y, 1,[], 1, N);
            T = reshape(T, 1,[], 1, N);

            Y = permute(Y, [1 3 2 4]);
            T = permute(T, [1 3 2 4]);

            R = size(Y,3);

            msError = sum(abs(Y-T).^2,3)/R;
    
            % Take mean over mini-batch.
            
            loss = sum(msError)/N;
        end
    end
end

