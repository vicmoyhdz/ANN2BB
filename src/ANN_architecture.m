% New ANN architecture by Victor Hernández (victorh@hi.is)
% University of Iceland - Politecnico di Milano
% July 2025. Updated May 2026.

function [varargout] = ANN_architecture(varargin)
%% *SET-UP*
ann = varargin{1};
nbs = varargin{2};
NoInputs=varargin{3};
NoOutputs=varargin{4};
dsg = varargin{5};
TransferLearning = varargin{6};
add_distance = varargin{7};
add_m = varargin{8};
add_lndistance = varargin{9};
add_vs30=varargin{10};
n_classes=varargin{11};
n_fm=varargin{12};
n_rg=varargin{13};
add_depth=varargin{14};
component=varargin{15};
eventID=varargin{16};

if strcmp(TransferLearning,'True')
    nbs2 = varargin{17};
end

iextra=0;
if strcmp(add_distance,'True')
    iextra=iextra+1;
end
if strcmp(add_m,'True')
    iextra=iextra+1;
end
if strcmp(add_lndistance,'True')
    iextra=iextra+1;
end
if strcmp(add_vs30,'True')
    iextra=iextra+1;
end
if strcmp(add_depth,'True')
    iextra=iextra+1;
end

n_first_level=round(ann.nhn*NoInputs);
if strcmp(component,'h12v')
    n_common=round(3*ann.nhn*NoOutputs);
else
    n_common=round(ann.nhn*NoOutputs);
end

%% *CREATE BASE NETWORK*
% number of Hidden Neurons
dsg.nhn = ann.nhn;

% Set up Division of Data for Training, Validation, Testing
dsg.net.divideParam.trainRatio = 72/100;
dsg.net.divideParam.valRatio   = 18/100;
dsg.net.divideParam.testRatio  =  10/100;

    [dsg.idx.trn,dsg.idx.vld,dsg.idx.tst] = trann_tv_sets(eventID,dsg.net.divideParam.valRatio,...
        dsg.net.divideParam.testRatio,1);
    % [dsg.idx.trn,dsg.idx.vld,dsg.idx.tst] = trann_tv_sets_eventwise(eventID,dsg.net.divideParam.valRatio,...
    %   dsg.net.divideParam.testRatio,1); %This is another alternative for event-wise split  

if strcmp(TransferLearning,'True')
        [idx2.trn,idx2.vld,idx2.tst] = trann_tv_sets(ones(nbs2,1),0.20,0.15,1);
end

%% Create branches

layers = dlnetwork;

if strcmp(component,'h12v')
    branches=3;
else
    branches=1;
end
%branch 1
input1=featureInputLayer(NoInputs,"Normalization","zscore",Name="input1");
Branch1 = [input1,fullyConnectedLayer(n_first_level, 'Name', 'fc_1'),...
    tanhLayer('Name', 'tanh_1')];
layers = addLayers(layers, Branch1);

%branches 2 & 5
if strcmp(component,'h12v')
    input2=featureInputLayer(NoInputs,"Normalization","zscore",Name="input2");
    Branch2 = [input2,fullyConnectedLayer(n_first_level, 'Name', 'fc_2'),...
    tanhLayer('Name', 'tanh_2')];

    input5=featureInputLayer(NoInputs,"Normalization","zscore",Name="input5");
    Branch5 = [input5,fullyConnectedLayer(n_first_level, 'Name', 'fc_5'),...
    tanhLayer('Name', 'tanh_5')];
    layers = addLayers(layers, Branch2); layers = addLayers(layers, Branch5);
end

if iextra>0
    %branch 3
    input3=featureInputLayer(iextra,"Normalization","zscore",Name="input3");
    extraBranch = [input3,fullyConnectedLayer(round(iextra*1.6), 'Name', 'fc_3'),...
    tanhLayer('Name', 'tanh_3')];
    branches=branches+1;
    layers = addLayers(layers, extraBranch);
end

if n_classes>0
    catInput = featureInputLayer(4, 'Name', 'categoryInput');
    catBranch = [ catInput,fullyConnectedLayer(5, 'Name', 'input4'),...
     reluLayer('Name', 'cat_relu1')];
    branches=branches+1;
    layers = addLayers(layers, catBranch);
end
if n_fm>0
    catInput_fm = featureInputLayer(3, 'Name', 'categoryInput_fm');
    catBranch_fm = [ catInput_fm,fullyConnectedLayer(5, 'Name', 'input6'),...
     reluLayer('Name', 'cat_relu2')];
    branches=branches+1;
    layers = addLayers(layers, catBranch_fm);
end
if n_rg>0
    catInput_rg = featureInputLayer(6, 'Name', 'categoryInput_rg');
    catBranch_rg = [ catInput_rg,fullyConnectedLayer(7, 'Name', 'input7'),...
     reluLayer('Name', 'cat_relu3')];
    branches=branches+1;
    layers = addLayers(layers, catBranch_rg);
end


%concatenation
if branches>1
    concat=concatenationLayer(1,branches,Name="concat");
    layers = addLayers(layers, concat);
end

%shared
shared = [fullyConnectedLayer(n_common, 'Name', 'fc_shared1'),...
    tanhLayer('Name', 'tanh_shared1')];
layers = addLayers(layers, shared);

% Outputs
output1 = fullyConnectedLayer(NoOutputs, 'Name', 'output1');
layers = addLayers(layers, output1);

if strcmp(component,'h12v')
    output2 = fullyConnectedLayer(NoOutputs, 'Name', 'output2');
    output3 = fullyConnectedLayer(NoOutputs, 'Name', 'output3');
    layers = addLayers(layers, output2); layers = addLayers(layers, output3);
end

%% Connect branches

if branches > 1

    iconcat = 1;

    % Main spectral branches
    layers = connectLayers(layers, 'tanh_1', sprintf('concat/in%d', iconcat));
    iconcat = iconcat + 1;

    if strcmp(component,'h12v')
        layers = connectLayers(layers, 'tanh_2', sprintf('concat/in%d', iconcat));
        iconcat = iconcat + 1;
        layers = connectLayers(layers, 'tanh_5', sprintf('concat/in%d', iconcat));
        iconcat = iconcat + 1;
    end

    % Extra continuous predictors
    if iextra > 0
        layers = connectLayers(layers, 'tanh_3', sprintf('concat/in%d', iconcat));
        iconcat = iconcat + 1;
    end

    % Site/class branch
    if n_classes > 0
        layers = connectLayers(layers, 'cat_relu1', sprintf('concat/in%d', iconcat));
        iconcat = iconcat + 1;
    end

    % Fault-mechanism branch
    if n_fm > 0
        layers = connectLayers(layers, 'cat_relu2', sprintf('concat/in%d', iconcat));
        iconcat = iconcat + 1;
    end

    % Region branch
    if n_rg > 0
        layers = connectLayers(layers, 'cat_relu3', sprintf('concat/in%d', iconcat));
        iconcat = iconcat + 1;
    end

    % Connect concatenation to shared block
    layers = connectLayers(layers, 'concat', 'fc_shared1');

else

    % Only one branch exists: input1 -> shared
    layers = connectLayers(layers, 'tanh_1', 'fc_shared1');

end

%Extra shared layer
% shared2 = [fullyConnectedLayer(n_common, 'Name', 'fc_shared2'),...
%     tanhLayer('Name', 'tanh_shared2')];
% layers = addLayers(layers, shared2);
% layers = connectLayers(layers, 'tanh_shared1', 'fc_shared2');

% Connect shared block to outputs
layers = connectLayers(layers, 'tanh_shared1', 'output1');

if strcmp(component,'h12v')
    layers = connectLayers(layers, 'tanh_shared1', 'output2');
    layers = connectLayers(layers, 'tanh_shared1', 'output3');
end

layers = initialize(layers);
numLearnables = sum(cellfun(@numel, layers.Learnables.Value));
fprintf('Total learnable parameters: %d\n', numLearnables);

figure; plot(layers)

%% *OUTPUT*
varargout{1} = dsg;
varargout{2} = layers;
if strcmp(TransferLearning,'True')
    varargout{3}=idx2;
else
    varargout{3}=dsg.idx; %not used but needed later
end
return
end
