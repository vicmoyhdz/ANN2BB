%% *GENERATION OF STRONG GROUND MOTION SIGNALS BY COUPLING PHYSICS-BASED ANALYSIS WITH ARTIFICIAL NEURAL NETWORKS*

%% *NOTES*
% _train_tv_sets_: function to select training and validation percentages
%% *N.B.*
% Need for:_randperm.m

function [varargout] = trann_tv_sets_eventwise(varargin)
%TRANN_TV_SETS_EVENTWISE Event-wise train/validation/test split.
%
% Usage:
%   [idx_trn, idx_vld, idx_tst] = trann_tv_sets_eventwise(eventID, pv, pt)
%   [idx_trn, idx_vld, idx_tst] = trann_tv_sets_eventwise(eventID, pv, pt, seed)
%
% Inputs:
%   eventID : [nr x 1] event identifier for each record
%   pv      : validation fraction, e.g. 0.18
%   pt      : test fraction, e.g. 0.10
%   seed    : optional random seed
%
% Outputs:
%   idx_trn, idx_vld, idx_tst : record indices
%
% Notes:
%   - All records from the same event are assigned to the same subset.
%   - The target fractions are record fractions, but exact values may differ
%     because events have different numbers of records.

    %% *SET-UP*
    eventID = varargin{1};
    pv      = varargin{2};
    pt      = varargin{3};

    if nargin >= 4 && ~isempty(varargin{4})
        rng(varargin{4});
    else
        rng("shuffle");
    end

    eventID = eventID(:);
    nr = numel(eventID);

    if pv < 0 || pt < 0 || (pv + pt) >= 1
        error('pv and pt must be non-negative and pv + pt < 1.');
    end

    %% *UNIQUE EVENTS*
    [evAll, ~, evIdx] = unique(eventID, 'stable');
    nev = numel(evAll);

    % Number of records per event
    nRecEv = accumarray(evIdx, 1, [nev 1]);

    %% *DEFINE TARGET RECORD COUNTS*
    nUse = floor(0.99 * nr);

    Q1 = ceil(nUse * pv);   % target validation records
    Q2 = ceil(nUse * pt);   % target test records

    %% *RANDOMIZE EVENTS*
    evPerm = randperm(nev);

    isEvVld = false(nev,1);
    isEvTst = false(nev,1);
    isEvTrn = false(nev,1);

    nVld = 0;
    nTst = 0;
    nAssigned = 0;

    %% *ASSIGN WHOLE EVENTS*
    for kk = 1:nev

        iEv = evPerm(kk);
        thisN = nRecEv(iEv);

        % Preserve the original behaviour of using about 95% of records.
        % Once nUse is reached, remaining events are ignored.
        if nAssigned >= nUse
            continue
        end

        deficitVld = Q1 - nVld;
        deficitTst = Q2 - nTst;

        if deficitVld > 0 || deficitTst > 0
            if deficitVld >= deficitTst && deficitVld > 0
                isEvVld(iEv) = true;
                nVld = nVld + thisN;
            elseif deficitTst > 0
                isEvTst(iEv) = true;
                nTst = nTst + thisN;
            else
                isEvTrn(iEv) = true;
            end
        else
            isEvTrn(iEv) = true;
        end

        nAssigned = nAssigned + thisN;
    end

    %% *GET RECORD INDICES*
    ev_vld = evAll(isEvVld);
    ev_tst = evAll(isEvTst);
    ev_trn = evAll(isEvTrn);

    idx.vld = find(ismember(eventID, ev_vld));
    idx.tst = find(ismember(eventID, ev_tst));
    idx.trn = find(ismember(eventID, ev_trn));

    %% *SAFETY CHECKS*
    if ~isempty(intersect(idx.trn, idx.vld))
        error('Training and validation sets overlap.');
    end

    if ~isempty(intersect(idx.trn, idx.tst))
        error('Training and test sets overlap.');
    end

    if ~isempty(intersect(idx.vld, idx.tst))
        error('Validation and test sets overlap.');
    end

    % Check no event appears in more than one subset
    if ~isempty(intersect(ev_trn, ev_vld)) || ...
       ~isempty(intersect(ev_trn, ev_tst)) || ...
       ~isempty(intersect(ev_vld, ev_tst))
        error('At least one event appears in more than one subset.');
    end

    %% *DISPLAY SUMMARY*
    fprintf('Requested approximately: TRN %.1f%%, VLD %.1f%%, TST %.1f%% of %.0f%% used records\n', ...
        100*(1-pv-pt), 100*pv, 100*pt, 99);

    fprintf('Actual record fractions: TRN %.1f%%, VLD %.1f%%, TST %.1f%%, unused %.1f%%\n', ...
        100*numel(idx.trn)/nr, ...
        100*numel(idx.vld)/nr, ...
        100*numel(idx.tst)/nr, ...
        100*(nr - numel(idx.trn) - numel(idx.vld) - numel(idx.tst))/nr);

    fprintf('Number of events:        TRN %d, VLD %d, TST %d\n', ...
        numel(ev_trn), numel(ev_vld), numel(ev_tst));

    %% *OUTPUT*
    varargout{1} = idx.trn;
    varargout{2} = idx.vld;
    varargout{3} = idx.tst;

    if nargout >= 4
        varargout{4} = ev_trn;
    end
    if nargout >= 5
        varargout{5} = ev_vld;
    end
    if nargout >= 6
        varargout{6} = ev_tst;
    end

end
