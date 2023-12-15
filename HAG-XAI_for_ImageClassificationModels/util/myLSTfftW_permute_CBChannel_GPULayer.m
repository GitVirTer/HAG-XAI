classdef myLSTfftW_permute_CBChannel_GPULayer < nnet.layer.Layer & nnet.layer.Formattable
    %PEEPHOLELSTMLAYER Peephole LSTM Layer

    properties
        % Layer properties.

        OutputMode
    end

    properties (Learnable)
        % Layer learnable parameters.

        W
    end

    methods
        function layer = myLSTfftW_permute_CBChannel_GPULayer(args)
            %PEEPHOLELSTMLAYER Peephole LSTM Layer
            %   layer = peepholeLSTMLayer(numHiddenUnits,inputSize)
            %   creates a peephole LSTM layer with the specified number of
            %   hidden units and input channels.
            %
            %   layer = peepholeLSTMLayer(numHiddenUnits,inputSize,Name=Value)
            %   creates a peephole LSTM layer and specifies additional
            %   options using one or more name-value arguments:
            %
            %      Name       - Name for the layer, specified as a string.
            %                   The default is "".
            %
            %      OutputMode - Output mode, specified as one of the
            %                   following:
            %                      "sequence" - Output the entire sequence
            %                                   of data.
            %
            %                      "last"     - Output the last time step
            %                                   of the data.
            %                   The default is "sequence".

            % Parse input arguments.
            arguments
                args.Name = "";
                args.OutputMode = "psd";
            end

            layer.Name = args.Name;
            layer.OutputMode = args.OutputMode;

            % Set layer description.
            layer.Description = "My LST Layer";

            % Initialize weights and bias.
            layer.W = initialWindow_ifft(256, 4, 35);
%             layer.W = initialFlatWindow(256, 4, 35);


        end
        
        function Z = predict(layer,X)
            %PREDICT Peephole LSTM predict function
            %   [Z,hiddenState,cellState] = predict(layer,X) forward
            %   propagates the data X through the layer and returns the
            %   layer output Z and the updated hidden and cell states. X
            %   is a dlarray with format "CBT" and Z is a dlarray with
            %   format "CB" or "CBT", depending on the layer OutputMode
            %   property.

            X = real(X);
            data = gpuArray(X);
            data = stripdims(data);
            data = permute(data, [3 2 1 4]);
            nCh = size(data,3);
            nSample = size(data,4);
            
            timeseries = reshape(data, [size(data,1) size(data,2) nCh*nSample]);
            
            sample_freq = 256;
            freqsamplingrate = 1;
            div = 8;
            freqRange = [4 35];
            % freqSamp = 2;
            
            freqSamplingRate = size(data,2)/sample_freq;
            minfreq = freqRange(1)*freqSamplingRate;
            maxfreq = freqRange(2)*freqSamplingRate;
            
            % **********************************************************************************
            timeseries_new = reshape(timeseries, [size(timeseries,2),size(timeseries,3)]);
            n = size(timeseries_new,1);
            nS = size(timeseries_new,2);
            
            vector_fft=fft(timeseries_new);
            vector_fft=cat(1,vector_fft,vector_fft);
            st = complex(zeros([ceil((maxfreq - minfreq+1)/freqsamplingrate), n, size(timeseries_new, 2)], 'like', timeseries_new));
            
            % if minfreq == 0
                data = repmat(mean(timeseries_new),[n 1]);
                data = reshape(data, [1 size(data,1) size(data,2)]);
                st(1,:,:) = data;
            % end
            
            for banana=freqsamplingrate:freqsamplingrate:(maxfreq-minfreq)
                gauss = repmat(layer.W(:,banana), [1 nS]);
                st(banana/freqsamplingrate+1,:,:)=ifft(vector_fft((minfreq+banana+1):(minfreq+banana+n), :).*fft(gauss));
            end
            
            % *****************************************************************************************
            st = gather(st);
            data_st_psd = st.*conj(st);
%             data_st_psd = log(data_st_psd+1e-5);
%             data_st_psd = cat(1,real(st),imag(st));
            
            SegN = size(timeseries,2)/div;
            st_psd_us = zeros([size(data_st_psd,1) SegN nCh*nSample], 'like', data);
            for nSeg = 1:SegN
                st_psd_us(:, nSeg, :) = permute(sum(permute(data_st_psd(:,(nSeg-1)*div+1 : nSeg*div,:), [2 1 3])), [2 1 3]);
            end

            if layer.OutputMode == "norm CB"
                st_psd_us = st_psd_us./max(max(st_psd_us));
            end
            
            Zp = zeros([size(data_st_psd,1) SegN nCh nSample], 'like', data);
            for iCh = 1:nCh
                Zp(:,:,iCh,:) = st_psd_us(:,:,iCh:nCh:end); % FTCB
            end

            if layer.OutputMode == "norm B"
                Zp = Zp./max(max(max(Zp)));
            end

            Zp = permute(Zp, [3 4 1 2]);    % CBFT
            Zp = Zp(:,:,:);
            Zp = permute(Zp, [3 2 1]);

            Z = gpuArray(dlarray(Zp, 'CBT'));
            
        end

    end
end

