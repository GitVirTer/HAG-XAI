classdef PCCRegressionLayer < nnet.layer.RegressionLayer ...
        & nnet.layer.Acceleratable
    % Example custom regression layer with mean-absolute-error loss.
    
    methods
        function layer = PCCRegressionLayer(name)
            % layer = maeRegressionLayer(name) creates a
            % mean-absolute-error regression layer and specifies the layer
            % name.
			
            % Set layer name.
            layer.Name = name;

            % Set layer description.
            layer.Description = 'Mean PCC';
        end
        
        function loss = forwardLoss(layer, Y, T)
            % loss = forwardLoss(layer, Y, T) returns the MAE loss between
            % the predictions Y and the training targets T.

            % Calculate Size.
            N = size(Y,2);

            % Flatten
            Y = reshape(Y, 1,[], 1, N);
            T = reshape(T, 1,[], 1, N);

            % Cal PCC
            epsX = 1e-5;
            xtd = Y-mean(Y,[1 2]);
            xtd_t = permute(xtd, [2 1 3 4]);
            ytd = T-mean(T,[1 2]);
            ytd_t = permute(ytd, [2 1 3 4]);
            PCC = pagemtimes(xtd, ytd_t)./(sqrt(pagemtimes(xtd, xtd_t)).*sqrt(pagemtimes(ytd, ytd_t))+epsX);

            % Cal Loss
            loss = squeeze(sum(1-PCC)/N);

        end
    end
end

