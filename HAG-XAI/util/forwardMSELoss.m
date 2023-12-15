    function [loss, msError] = forwardMSELoss(Y, T)
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