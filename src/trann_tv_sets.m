%% *GENERATION OF STRONG GROUND MOTION SIGNALS BY COUPLING PHYSICS-BASED ANALYSIS WITH ARTIFICIAL NEURAL NETWORKS*

%% *NOTES*
% _train_tv_sets_: function to select training and validation percentages
%% *N.B.*
% Need for:_randperm.m_

function [varargout] = trann_tv_sets(varargin)

 if nargin >= 5 && ~isempty(varargin{5})
        rng(varargin{5});
 else
        rng("shuffle");
 end
    %% *SET-UP*
    nr   = length(varargin{1});
    pv   = varargin{2};
    pt   = varargin{3};
    percentage_records=varargin{4};
    
    %% *DEFINE PERCENTAGES*
    Q1   = ceil(percentage_records*nr*pv);
    Q2   = ceil(percentage_records*nr*pt);
    Q3   = floor(percentage_records*nr)-Q1-Q2;
    
    idx.all(:,1) = randperm(nr);
    idx.vld      = idx.all(1:Q1,1);
    idx.tst      = idx.all(Q1+(1:Q2),1);
    idx.trn      = idx.all(Q2+(1:Q3),1);
    
    %% *OUTPUT*
    varargout{1} = idx.trn;
    varargout{2} = idx.vld;
    varargout{3} = idx.tst;
    
    return
end
