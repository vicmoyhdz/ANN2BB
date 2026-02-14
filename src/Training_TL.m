    
% ------------------------------------------------

% Start from pretrained net (BEFORE fine-tuning) as reference theta0
dlnet0 = NNs{i_}.net;     % pretrained weights (reference)
dlnet  = NNs{i_}.net;     % initialize fine-tune net from same weights

% Align theta0 to dlnet learnables (by Layer/Parameter key)
theta0 = alignTheta0(dlnet, dlnet0);

% Mask: which learnables are allowed to change (only in layerName)
L = dlnet.Learnables;
maskFT = ismember(string(L.Layer), layerName);

% Build minibatchqueues directly from already-built combined datastores
[nAll, nX] = inferCombinedArity(dsTrn2);   % nAll = #streams = nX inputs + 3 targets
fmt = repmat("CB", 1, nAll);

mbqTrn = minibatchqueue(dsTrn2, ...
    "MiniBatchSize", miniBatchSize, ...
    "MiniBatchFcn",  @mbatchFcnAnyBranches, ...
    "MiniBatchFormat", fmt, ...
    "PartialMiniBatch","discard");

mbqVld = minibatchqueue(dsVld2, ...
    "MiniBatchSize", miniBatchSize, ...
    "MiniBatchFcn",  @mbatchFcnAnyBranches, ...
    "MiniBatchFormat", fmt, ...
    "PartialMiniBatch","discard");

% Adam state
trailingAvg = [];
trailingAvgSq = [];
iteration = 0;

bestVal = inf;
bestNet = dlnet;
bad = 0;

for epoch = 1:maxEpochs
    shuffle(mbqTrn);

    while hasdata(mbqTrn)
        iteration = iteration + 1;

        data = cell(1,nAll);
        [data{:}] = next(mbqTrn);

        X = data(1:nX);
        T = data(nX+1:end); % last 3

        [loss, grads] = dlfeval(@modelGradientsL2SP, dlnet, X, T, lossFcn, theta0, maskFT, lambdaSP);

        % Freeze everything else
        grads = zeroFrozenGradients(grads, dlnet.Learnables, maskFT);

        % Clip global L2-norm
        grads = clipGradientsL2(grads, gradThresh);

        % Adam update
        [dlnet, trailingAvg, trailingAvgSq] = adamupdate(dlnet, grads, ...
            trailingAvg, trailingAvgSq, iteration, learnRate);
    end

    % Validation
    valLoss = computeValLoss(dlnet, mbqVld, lossFcn, nAll, nX);

    if valLoss < bestVal
        bestVal = valLoss;
        bestNet = dlnet;
        bad = 0;
    else
        bad = bad + 1;
        if bad >= patience
            break
        end
    end
end

NNs{i_}.net = bestNet;  % fine-tuned net with L2-SP


%% Helpers

function [nAll, nX] = inferCombinedArity(ds)
    tmp = read(ds);   % combined datastore returns a cell array
    reset(ds);
    nAll = numel(tmp);
    if nAll < 4
        error("Combined datastore must have at least 1 input stream + 3 targets.");
    end
    nX = nAll - 3;
end

function varargout = mbatchFcnAnyBranches(varargin)
% Each varargin{i} can be numeric [B x F] OR cell containing numeric rows.
% Output dlarray [F x B] with format "CB".

    nAll = numel(varargin);
    out  = cell(1,nAll);

    for i = 1:nAll
        A = varargin{i};

        % ---- convert possible cell batch to numeric ----
        if iscell(A)
            if isscalar(A) && isnumeric(A{1})
                % 1x1 cell containing the full batch matrix
                A = A{1};
            else
                % Bx1 cell of rows (or similar)
                try
                    A = cell2mat(A);
                catch
                    A = cat(1, A{:});
                end
            end
        end

        A = single(A);          % now must be numeric
        out{i} = dlarray(A', "CB");   % [F x B]
    end

    varargout = out;
end


function theta0 = alignTheta0(dlnet, dlnet0)
    L  = dlnet.Learnables;
    L0 = dlnet0.Learnables;

    key  = string(L.Layer ) + "/" + string(L.Parameter );
    key0 = string(L0.Layer) + "/" + string(L0.Parameter);

    [idxIn0, tf] = ismember(key, key0);
    if ~all(tf)
        error("Architectures mismatch: some learnables not found in pretrained net.");
    end

    theta0 = L0.Value(idxIn0);   % cell array aligned to L rows
end

function [loss, grads] = modelGradientsL2SP(dlnet, X, T, lossFcn, theta0, maskFT, lambdaSP)

    % Forward with variable number of inputs
    [Y1,Y2,Y3] = forward(dlnet, X{:});

    % Your existing data loss
    dataLoss = lossFcn(Y1,Y2,Y3, T{1},T{2},T{3});

    % L2-SP penalty: sum ||theta - theta0||^2 over selected layers
    L = dlnet.Learnables;

    spSum = dlarray(single(0));
    nElem = 0;

    for i = 1:height(L)
        if ~maskFT(i), continue; end
        Wi  = L.Value{i};
        W0i = theta0{i};
        dW  = Wi - W0i;

        spSum = spSum + sum(dW.^2, "all");
        nElem = nElem + numel(extractdata(Wi));
    end

    spPenalty = spSum / max(nElem,1);
    loss = dataLoss + lambdaSP * spPenalty;

    grads = dlgradient(loss, dlnet.Learnables);

    % curv = Y2(3:end,:) - 2*Y2(2:end-1,:) + Y2(1:end-2,:);
    % Ps   = mean(curv.^2,"all");          % curvature penalty
    % ratio = (lambdaS*Ps) / dataLoss; 

    % ratioSP = (lambdaSP * spPenalty) / max(dataLoss,1e-12)

end

function grads = zeroFrozenGradients(grads, learnables, maskFT)
    for i = 1:height(grads)
        if ~maskFT(i)
            grads.Value{i} = zeros(size(learnables.Value{i}), "like", learnables.Value{i});
        end
    end
end

function grads = clipGradientsL2(grads, thresh)
    s = 0;
    for i = 1:height(grads)
        g = grads.Value{i};
        s = s + sum(g.^2,"all");
    end
    nrm = sqrt(s);

    if extractdata(nrm) > thresh
        scale = thresh / nrm;
        for i = 1:height(grads)
            grads.Value{i} = grads.Value{i} * scale;
        end
    end
end


function valLoss = computeValLoss(dlnet, mbqVld, lossFcn, nAll, nX)
    reset(mbqVld);
    Ls = [];

    while hasdata(mbqVld)
        data = cell(1,nAll);
        [data{:}] = next(mbqVld);

        X = data(1:nX);
        T = data(nX+1:end);

        [Y1,Y2,Y3] = forward(dlnet, X{:});
        L = lossFcn(Y1,Y2,Y3, T{1},T{2},T{3});
        Ls(end+1) = gather(extractdata(L)); %#ok<AGROW>
    end

    valLoss = mean(Ls);
end



