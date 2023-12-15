function [loss, loss_pcc] = forwardPCCLoss(Y, T)
            % loss = forwardLoss(layer, Y, T) returns the MAE loss between
            % the predictions Y and the training targets T.

            % Calculate Size.
            R = size(Y,3);
            N = size(Y,4);

            % Flatten
            Y = reshape(Y, 1,[], R, N);
            T = reshape(T, 1,[], R, N);

            % Cal PCC
            epsX = 1e-6;
            xtd = Y-mean(Y,[1 2]);
            xtd_t = permute(xtd, [2 1 3 4]);
            ytd = T-mean(T,[1 2]);
            ytd_t = permute(ytd, [2 1 3 4]);
            PCC = pagemtimes(xtd, ytd_t)./(sqrt(pagemtimes(xtd, xtd_t)).*sqrt(pagemtimes(ytd, ytd_t))+epsX);

            % Cal Loss
            loss_pcc = 1-PCC;
            loss = sum(loss_pcc)/N;

end