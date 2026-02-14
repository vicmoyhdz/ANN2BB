%% TRAINING ANN TO PREDICT SHORT-PERIOD SA FROM LONG-PERIOD AND SCALAR METADATA
% Victor M. Hernández-Aguirre (victorh@hi.is)
% University of Iceland - Politecnico di Milano
% January 2025. Updated January 2026

clc
clear
close all
 
addpath src\ %subfolder with routines
folder_save='ANNs'; %folder to save the ouput ANNs

%% TRAIN SET-UP (CUSTOMIZE)

TransferLearning = 'False'; %if True you need to define also dbn_name2, otherwise dbn_name2 is ignored
dbn_name = 'ESM_NGA_130';
dbn_name2 = 'DATABASE_TL'; %Database for transfer learning (TL). Put the same if not using TL
%The data sets should be located inside subfolder database

% DEFINE THE NUMBER OF NETS TO BE TRAINED
num_nets = 1; %number of individual nets
n_LoopsANN = 1; %number of trained nets before choosing the best one
add_distance = 'True';
add_m = 'True';
add_lndistance = 'True';
add_vs30 = 'True';
add_depth = 'True';

separate_classes = 'False';
add_fm = 'True';
separate_regions = 'True';

%% DEFINE TRAIN METADATA (CUSTOMIZE)

% ANN METADATA
% number of ANNs
ann.trn.nr = 2; 
% corner period
TnC = [1];
% direction (h12v=Three-components;ud=vertical;h=rotational-invariant h_inv)
cp  = {'h12v'};
nnr = 1; %scaling factor N1=nnr*No. input periods, N2=3*nnr*No. output periods
%;vTn = Vector with the periods at which the spectral accelerations of the
%database are computed. For the given databases do not change
vTn = [0;0.01;0.025;0.04;0.05;0.07;(0.1:0.05:0.5)';0.6;0.7;0.75;0.8;0.9;(1:0.2:2)';(2.5:0.5:5)';(6:1:10)'];

%%

% main workdir
wd = strcat(cd,'\',folder_save);
% save path
dbn = strcat(cd,'\database\',dbn_name,'.mat');
dbn2 = strcat(cd,'\database\',dbn_name2,'.mat');

if exist(wd,'dir')~=7
    wd = strcat(cd,'\',folder_save);   
    dbn = strcat(cd,'\database\',dbn_name,'.mat');
    dbn2 = strcat(cd,'\database\',dbn_name2,'.mat');
end

ann.trn.wd = fullfile(wd);
fprintf('Training Workdir: %s\n',ann.trn.wd);
fprintf('Training Database: %s\n',dbn);

for i_=1:ann.trn.nr
    ann.trn.mtd(i_).TnC = TnC(i_);
    ann.trn.mtd(i_).cp  = cp{i_};
    ann.trn.mtd(i_).nhn = nnr;
end
clear TnC cp 
% database file names
for i_ = 1:ann.trn.nr
    ann.trn.mtd(i_).dbn = dbn;
end

%% Training
tstart=tic;
for iNet = 1:num_nets    
    net_ID = iNet;
    for i_ = 1:ann.trn.nr
        train_ann_PSA(ann.trn.wd,ann.trn.mtd(i_),dbn_name,net_ID,n_LoopsANN,TransferLearning,...
        dbn2,add_distance,add_m,add_lndistance,separate_classes,add_vs30,add_fm,separate_regions,add_depth,vTn);
    end
end
tEnd_all = toc(tstart);
fprintf('\n\t ..END TRAINING.. in %s sec.\n',num2str(tEnd_all));