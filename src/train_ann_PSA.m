%% TRAINING ANN TO PREDICT SHORT-PERIOD SA FROM LONG-PERIOD AND SCALAR METADATA
% By Victor M. Hernández-Aguirre (victorh@hi.is) 
% University of Iceland - Politecnico di Milano
% January 2025. Updated January 2026
function train_ann_PSA(varargin)
%% *SET-UP*
wd  = varargin{1};
ann = varargin{2};
dbn_name = varargin{3};
net_ID  = varargin{4};
n_LoopsANN = varargin{5};
TransferLearning = varargin{6};
dbn2 = varargin{7};
add_distance = varargin{8};
add_m = varargin{9};
add_lndistance = varargin{10};
separate_classes = varargin{11};
add_vs30 = varargin{12};
add_fm = varargin{13};
separate_regions = varargin{14};
add_depth = varargin{15};

if net_ID <10
    verNet = ['v0',num2str(net_ID)];
elseif net_ID <100
    verNet = ['v',num2str(net_ID)];
end

% load database
db = load(ann.dbn);
db.nr  = size(db.DATABASE,2);
db.vTn = varargin{16};
db.nT  = numel(db.vTn);

% define input/target natural periods
all_periods=[0,0.05,0.07,(0.1:0.05:0.5),0.6,0.7,0.75,0.8,0.9,1.0:0.2:2.0,(2.5:0.5:5)];
% all_periods = [0,0.01,0.025,0.04,0.05,0.07,(0.1:0.05:0.5),0.6,0.7,0.75,0.8,0.9,(1:0.2:2),(2.5:0.5:5),(6:1:10)];
[inp.vTn,tar.vTn,inp.nT,tar.nT] = trann_define_inout(ann.TnC,all_periods);
% check input/target natural periods with database
[inp.idx,tar.idx] = trann_check_vTn(inp,tar,db,1e-6);

if strcmp(TransferLearning,'True')
    db2 = load(dbn2);
    db2.nr  = size(db2.DATABASE,2);
    db2.vTn = varargin{16};
    db2.nT  = numel(db2.vTn);
    [inp2.vTn,tar2.vTn,inp2.nT,tar2.nT] = trann_define_inout(ann.TnC,all_periods);
    [inp2.idx,tar2.idx] = trann_check_vTn(inp2,tar2,db2,1e-6);
end

%% EXTRACTING ANN INPUTS/TARGETS FROM DATABASE STRUCTURE

switch ann.cp
    % Three COMPONENTS (separate branches)
    case {'h12v'}
        for j_ = 1:db.nr
            PSA_1(j_,:) = db.DATABASE(j_).psa_h1(:)';
            PSA_2(j_,:) = db.DATABASE(j_).psa_h2(:)';
            PSA_3(j_,:) = db.DATABASE(j_).psa_v(:)';
        end
        if strcmp(TransferLearning,'True')
            for j_ = 1:db2.nr
                PSA2_1(j_,:) = db2.DATABASE(j_).psa_h1(:)';
                PSA2_2(j_,:) = db2.DATABASE(j_).psa_h2(:)';
                PSA2_3(j_,:) = db2.DATABASE(j_).psa_v(:)';
            end
        end
        % HORIZONTAL ROTATIONAL INVARIANT
    case {'h_inv'}
        for j_ = 1:db.nr
            PSA_1(j_,:) = [db.DATABASE(j_).psa_inv(:)'];
            % PGV(j_,:) = db.DATABASE(j_).pgv(1);
        end
        if strcmp(TransferLearning,'True')
            for j_ = 1:db2.nr
                PSA2_1(j_,:) = [db2.DATABASE(j_).psa_inv(:)'];
            end
        end
        % HORIZONTAL COMPONENT 1
    case {'h1'}
        for j_ = 1:db.nr
            PSA_1(j_,:) = [db.DATABASE(j_).psa_inv(:)'];
        end
        if strcmp(TransferLearning,'True')
            for j_ = 1:db2.nr
                PSA2_1(j_,:) = [db2.DATABASE(j_).psa_inv(:)'];
            end
        end
        % HORIZONTAL COMPONENT 2
    case {'h2'}
        for j_ = 1:db.nr
            PSA(j_,:) = db.DATABASE(j_).psa_h2(:)';
        end
        if strcmp(TransferLearning,'True')
            for j_ = 1:db2.nr
                PSA2(j_,:) = db2.DATABASE(j_).psa_h2(:)';
            end
        end
    case 'gh'
        for j_ = 1:db.nr
            PSA_1(j_,:) = geomean([db.DATABASE(1,j_).psa_h1(:)';...
                db.DATABASE(1,j_).psa_h2(:)'],1);
        end
        if strcmp(TransferLearning,'True')
            for j_ = 1:db2.nr
                PSA2_1(j_,:) = geomean([db2.DATABASE(1,j_).psa_h1(:)';...
                    db2.DATABASE(1,j_).psa_h2(:)'],1);
            end
        end
        % VERTICAL COMPONENT
    case 'ud'
        for j_ = 1:db.nr
            PSA_1(j_,:) = db.DATABASE(j_).psa_v(:)';
        end
        if strcmp(TransferLearning,'True')
            for j_ = 1:db2.nr
                PSA2_1(j_,:) = db2.DATABASE(j_).psa_v(:)';
            end
        end
end

%% *DEFINE INPUT/TARGET PSA POOL (LOG)*

if strcmp(ann.cp,'h12v')
    inp.DATABASE_1  = -999*ones(inp.nT,db.nr);
    tar.DATABASE_1  = -999*ones(tar.nT,db.nr);
    inp.DATABASE_2  = -999*ones(inp.nT,db.nr);
    tar.DATABASE_2 = -999*ones(tar.nT,db.nr);
    inp.DATABASE_5  = -999*ones(inp.nT,db.nr);
    tar.DATABASE_5 = -999*ones(tar.nT,db.nr);

    if strcmp(TransferLearning,'True')
        inp2.DATABASE_1  = -999*ones(inp.nT,db2.nr);
        tar2.DATABASE_1  = -999*ones(tar.nT,db2.nr);
        inp2.DATABASE_2  = -999*ones(inp.nT,db2.nr);
        tar2.DATABASE_2  = -999*ones(tar.nT,db2.nr);
        inp2.DATABASE_5  = -999*ones(inp.nT,db2.nr);
        tar2.DATABASE_5  = -999*ones(tar.nT,db2.nr);
    end
else
    inp.DATABASE_1  = -999*ones(inp.nT,db.nr);
    tar.DATABASE_1  = -999*ones(tar.nT,db.nr);
    if strcmp(TransferLearning,'True')
        inp2.DATABASE_1  = -999*ones(inp.nT,db2.nr);
        tar2.DATABASE_1  = -999*ones(tar.nT,db2.nr);
    end
end

for i_=1:inp.nT

    inp.DATABASE_1(i_,1:db.nr) = log(PSA_1(1:db.nr,inp.idx(i_))./100)';
    if strcmp(TransferLearning,'True')
        inp2.DATABASE_1(i_,1:db2.nr) = log(PSA2_1(1:db2.nr,inp2.idx(i_))./100)';
    end
    if strcmp(ann.cp,'h12v')
        inp.DATABASE_2(i_,1:db.nr) = log(PSA_2(1:db.nr,inp.idx(i_))./100)';
        inp.DATABASE_5(i_,1:db.nr) = log(PSA_3(1:db.nr,inp.idx(i_))./100)';
        if strcmp(TransferLearning,'True')
            inp2.DATABASE_2(i_,1:db2.nr) = log(PSA2_2(1:db2.nr,inp2.idx(i_))./100)';
            inp2.DATABASE_5(i_,1:db2.nr) = log(PSA2_3(1:db2.nr,inp2.idx(i_))./100)';
        end
    end
end

for i_=1:tar.nT

    tar.DATABASE_1(i_,1:db.nr) = log(PSA_1(1:db.nr,tar.idx(i_))./100)';
    if strcmp(TransferLearning,'True')
        tar2.DATABASE_1(i_,1:db2.nr) = log(PSA2_1(1:db2.nr,tar2.idx(i_))./100)';
    end

    if strcmp(ann.cp,'h12v')
        tar.DATABASE_2(i_,1:db.nr) = log(PSA_2(1:db.nr,tar.idx(i_))./100)';
        tar.DATABASE_3(i_,1:db.nr) = log(PSA_3(1:db.nr,tar.idx(i_))./100)';
        if strcmp(TransferLearning,'True')
            tar2.DATABASE_2(i_,1:db2.nr) = log(PSA2_2(1:db2.nr,tar2.idx(i_))./100)';
            tar2.DATABASE_3(i_,1:db2.nr) = log(PSA2_3(1:db2.nr,tar2.idx(i_))./100)';
        end
    end
end

%% Add extra inputs according to flags
index_extra=0;

if strcmp(add_distance,'True')
    index_extra=index_extra+1;
    for j_ = 1:db.nr
        inp.DATABASE_3(index_extra,j_) =  (max(db.DATABASE(j_).Rjb,0.01));
    end
    if strcmp(TransferLearning,'True')
        for j_ = 1:db2.nr
            inp2.DATABASE_3(index_extra,j_) =  (max(db2.DATABASE(j_).Rjb,0.01));
        end
    end
end

if strcmp(add_m,'True')
    index_extra=index_extra+1;
    for j_ = 1:db.nr
        inp.DATABASE_3(index_extra,j_) = db.DATABASE(j_).Mw;
    end
    if strcmp(TransferLearning,'True')
        for j_ = 1:db2.nr
            inp2.DATABASE_3(index_extra,j_) = db2.DATABASE(j_).Mw;
        end
    end
end

if strcmp(add_lndistance,'True')
    index_extra=index_extra+1;
    for j_ = 1:db.nr
        inp.DATABASE_3(index_extra,j_) =  log(max(db.DATABASE(j_).Rjb,0.01));
    end
    if strcmp(TransferLearning,'True')
        for j_ = 1:db2.nr
            inp2.DATABASE_3(index_extra,j_) =  log(max(db2.DATABASE(j_).Rjb,0.01));
        end
    end
end

if strcmp(add_vs30,'True')
    index_extra=index_extra+1;
    for j_ = 1:db.nr
        inp.DATABASE_3(index_extra,j_) =  log(db.DATABASE(j_).Vs30);
    end
    if strcmp(TransferLearning,'True')
        for j_ = 1:db2.nr
            inp2.DATABASE_3(index_extra,j_) =  log(db2.DATABASE(j_).Vs30);
        end
    end
end

if strcmp(add_depth,'True')
    index_extra=index_extra+1;
    for j_ = 1:db.nr
        inp.DATABASE_3(index_extra,j_) =  db.DATABASE(j_).event_depth;
    end
    if strcmp(TransferLearning,'True')
        for j_ = 1:db2.nr
            inp2.DATABASE_3(index_extra,j_) =  db2.DATABASE(j_).event_depth;
        end
    end
end

n_classes=0;
if strcmp(separate_classes,'True')
    Class = arrayfun(@(s) string(s.site_EC8), db.DATABASE(:));
    Class = replace(Class, ["A*","B*","C*","D*","E"], ["A","B","C","D","B"]);
    Class = categorical(Class);
    inp.DATABASE_4 = onehotencode(Class',1);
    n_classes = size(inp.DATABASE_4,1);
    if strcmp(TransferLearning,'True')
        Class2 = arrayfun(@(s) string(s.site_EC8), db2.DATABASE(:));
        Class2 = replace(Class2, ["A*","B*","C*","D*","E"], ["A","B","C","D","B"]);
        Class2 = categorical(Class2);
        inp2.DATABASE_4 = onehotencode(Class2',1);
    end
end

n_fm=0;
if strcmp(add_fm,'True')
    fault_mech=[db.DATABASE.fm_type];
    inp.DATABASE_6 = onehotencode(categorical(fault_mech,["NF","TF","SS"]),1);
    n_fm = size(inp.DATABASE_6,1);
    if strcmp(TransferLearning,'True')
        fault_mech2=[db2.DATABASE.fm_type];
        inp2.DATABASE_6 = onehotencode(categorical(fault_mech2,["NF","TF","SS"]),1);
    end
end

n_rg=0;
if strcmp(separate_regions,'True')
    special = ["IT","US","TW","TR","JP"];
    areas=[db.DATABASE.area];
    % areas(strcmp(areas,"IR"))="TR";
    areas(~ismember(areas, special)) = "Other";
    area_cat = categorical(areas, [special "Other"]);
    inp.DATABASE_7 = onehotencode(area_cat,1);
    n_rg = size(inp.DATABASE_7,1);
    if strcmp(TransferLearning,'True')
        areas2=[db2.DATABASE.area];
        % areas(strcmp(areas,"IR"))="TR";
        areas2(~ismember(areas2, special)) = "Other";
        area_cat2 = categorical(areas2, [special "Other"]);
        inp2.DATABASE_7 = onehotencode(area_cat2,1);
    end
end

dsg.ntr=n_LoopsANN;
NNs = cell(dsg.ntr,1);
prf.vld = -999*ones(dsg.ntr,1);
out.prf = 0.0;

if ~strcmp(TransferLearning,'True')
    db2=db; %not used but needed
end

for i_=1:dsg.ntr
    % Creating ANN architecture based on input variables
    [dsg,layers,idx2] = ANN_architecture(ann,db.nr,inp.nT,tar.nT,dsg,TransferLearning,add_distance,add_m,add_lndistance,add_vs30,n_classes,n_fm,n_rg,add_depth,ann.cp,db2.nr);

    fprintf('ANN %u/%u: \n',i_,dsg.ntr);

    NNs{i_}.idx=dsg.idx;
    if strcmp(TransferLearning,'True')
     NNs{i_}.idx_TL=idx2;
    end

    %% Saving INPUTS/TARGETS as datastore format needed for training
    if strcmp(ann.cp,'h12v')
        if index_extra>0 && n_classes>0 && n_fm>0 && n_rg>0
            NNs{i_}.inp.trn = {inp.DATABASE_1(:,dsg.idx.trn)',inp.DATABASE_2(:,dsg.idx.trn)',inp.DATABASE_5(:,dsg.idx.trn)',inp.DATABASE_3(:,dsg.idx.trn)',inp.DATABASE_4(:,dsg.idx.trn)',inp.DATABASE_6(:,dsg.idx.trn)',inp.DATABASE_7(:,dsg.idx.trn)'};
            NNs{i_}.tar.trn = {tar.DATABASE_1(:,dsg.idx.trn)',tar.DATABASE_2(:,dsg.idx.trn)',tar.DATABASE_3(:,dsg.idx.trn)'};
            NNs{i_}.inp.vld = {inp.DATABASE_1(:,dsg.idx.vld)',inp.DATABASE_2(:,dsg.idx.vld)',inp.DATABASE_5(:,dsg.idx.vld)',inp.DATABASE_3(:,dsg.idx.vld)',inp.DATABASE_4(:,dsg.idx.vld)',inp.DATABASE_6(:,dsg.idx.vld)',inp.DATABASE_7(:,dsg.idx.vld)'};
            NNs{i_}.tar.vld = {tar.DATABASE_1(:,dsg.idx.vld)',tar.DATABASE_2(:,dsg.idx.vld)',tar.DATABASE_3(:,dsg.idx.vld)'};
            NNs{i_}.inp.tst = {inp.DATABASE_1(:,dsg.idx.tst)',inp.DATABASE_2(:,dsg.idx.tst)',inp.DATABASE_5(:,dsg.idx.tst)',inp.DATABASE_3(:,dsg.idx.tst)',inp.DATABASE_4(:,dsg.idx.tst)',inp.DATABASE_6(:,dsg.idx.tst)',inp.DATABASE_7(:,dsg.idx.tst)'};
            NNs{i_}.tar.tst = {tar.DATABASE_1(:,dsg.idx.tst)',tar.DATABASE_2(:,dsg.idx.tst)',tar.DATABASE_3(:,dsg.idx.tst)'};

            dsX1Trn_1 = arrayDatastore(inp.DATABASE_1(:,dsg.idx.trn)');
            dsX1Trn_2 = arrayDatastore(inp.DATABASE_2(:,dsg.idx.trn)');
            dsX1Trn_5 = arrayDatastore(inp.DATABASE_5(:,dsg.idx.trn)');
            dsX1Trn_3 = arrayDatastore(inp.DATABASE_3(:,dsg.idx.trn)');
            dsX1Trn_4 = arrayDatastore(inp.DATABASE_4(:,dsg.idx.trn)');
            dsX1Trn_6 = arrayDatastore(inp.DATABASE_6(:,dsg.idx.trn)');
            dsX1Trn_7 = arrayDatastore(inp.DATABASE_7(:,dsg.idx.trn)');
            dsT1Trn_1 = arrayDatastore(tar.DATABASE_1(:,dsg.idx.trn)');
            dsT1Trn_2 = arrayDatastore(tar.DATABASE_2(:,dsg.idx.trn)');
            dsT1Trn_3 = arrayDatastore(tar.DATABASE_3(:,dsg.idx.trn)');
            dsTrn1 = combine(dsX1Trn_1,dsX1Trn_2,dsX1Trn_5,dsX1Trn_3,dsX1Trn_4,dsX1Trn_6,dsX1Trn_7,dsT1Trn_1,dsT1Trn_2,dsT1Trn_3);

            dsX1vld_1 = arrayDatastore(inp.DATABASE_1(:,dsg.idx.vld)');
            dsX1vld_2 = arrayDatastore(inp.DATABASE_2(:,dsg.idx.vld)');
            dsX1vld_5 = arrayDatastore(inp.DATABASE_5(:,dsg.idx.vld)');
            dsX1vld_3 = arrayDatastore(inp.DATABASE_3(:,dsg.idx.vld)');
            dsX1vld_4 = arrayDatastore(inp.DATABASE_4(:,dsg.idx.vld)');
            dsX1vld_6 = arrayDatastore(inp.DATABASE_6(:,dsg.idx.vld)');
            dsX1vld_7 = arrayDatastore(inp.DATABASE_7(:,dsg.idx.vld)');
            dsT1vld_1 = arrayDatastore(tar.DATABASE_1(:,dsg.idx.vld)');
            dsT1vld_2 = arrayDatastore(tar.DATABASE_2(:,dsg.idx.vld)');
            dsT1vld_3 = arrayDatastore(tar.DATABASE_3(:,dsg.idx.vld)');
            dsVld1= combine(dsX1vld_1,dsX1vld_2,dsX1vld_5,dsX1vld_3,dsX1vld_4,dsX1vld_6,dsX1vld_7,dsT1vld_1,dsT1vld_2,dsT1vld_3);

            if strcmp(TransferLearning,'True')
                NNs{i_}.inp2.trn = {inp2.DATABASE_1(:,idx2.trn)',inp2.DATABASE_2(:,idx2.trn)',inp2.DATABASE_5(:,idx2.trn)',inp2.DATABASE_3(:,idx2.trn)',inp2.DATABASE_4(:,idx2.trn)',inp2.DATABASE_6(:,idx2.trn)',inp2.DATABASE_7(:,idx2.trn)'};
                NNs{i_}.tar2.trn = {tar2.DATABASE_1(:,idx2.trn)',tar2.DATABASE_2(:,idx2.trn)',tar2.DATABASE_3(:,idx2.trn)'};
                NNs{i_}.inp2.vld = {inp2.DATABASE_1(:,idx2.vld)',inp2.DATABASE_2(:,idx2.vld)',inp2.DATABASE_5(:,idx2.vld)',inp2.DATABASE_3(:,idx2.vld)',inp2.DATABASE_4(:,idx2.vld)',inp2.DATABASE_6(:,idx2.vld)',inp2.DATABASE_7(:,idx2.vld)'};
                NNs{i_}.tar2.vld = {tar2.DATABASE_1(:,idx2.vld)',tar2.DATABASE_2(:,idx2.vld)',tar2.DATABASE_3(:,idx2.vld)'};
                NNs{i_}.inp2.tst = {inp2.DATABASE_1(:,idx2.tst)',inp2.DATABASE_2(:,idx2.tst)',inp2.DATABASE_5(:,idx2.tst)',inp2.DATABASE_3(:,idx2.tst)',inp2.DATABASE_4(:,idx2.tst)',inp2.DATABASE_6(:,idx2.tst)',inp2.DATABASE_7(:,idx2.tst)'};
                NNs{i_}.tar2.tst = {tar2.DATABASE_1(:,idx2.tst)',tar2.DATABASE_2(:,idx2.tst)',tar2.DATABASE_3(:,idx2.tst)'};


                dsX2Trn_1 = arrayDatastore(inp2.DATABASE_1(:,idx2.trn)');
                dsX2Trn_2 = arrayDatastore(inp2.DATABASE_2(:,idx2.trn)');
                dsX2Trn_5 = arrayDatastore(inp2.DATABASE_5(:,idx2.trn)');
                dsX2Trn_3 = arrayDatastore(inp2.DATABASE_3(:,idx2.trn)');
                dsX2Trn_4 = arrayDatastore(inp2.DATABASE_4(:,idx2.trn)');
                dsX2Trn_6 = arrayDatastore(inp2.DATABASE_6(:,idx2.trn)');
                dsX2Trn_7 = arrayDatastore(inp2.DATABASE_7(:,idx2.trn)');
                dsT2Trn_1 = arrayDatastore(tar2.DATABASE_1(:,idx2.trn)');
                dsT2Trn_2 = arrayDatastore(tar2.DATABASE_2(:,idx2.trn)');
                dsT2Trn_3 = arrayDatastore(tar2.DATABASE_3(:,idx2.trn)');
                dsTrn2 = combine(dsX2Trn_1,dsX2Trn_2,dsX2Trn_5,dsX2Trn_3,dsX2Trn_4,dsX2Trn_6,dsX2Trn_7,dsT2Trn_1,dsT2Trn_2,dsT2Trn_3);

                dsX2vld_1 = arrayDatastore(inp2.DATABASE_1(:,idx2.vld)');
                dsX2vld_2 = arrayDatastore(inp2.DATABASE_2(:,idx2.vld)');
                dsX2vld_5 = arrayDatastore(inp2.DATABASE_5(:,idx2.vld)');
                dsX2vld_3 = arrayDatastore(inp2.DATABASE_3(:,idx2.vld)');
                dsX2vld_4 = arrayDatastore(inp2.DATABASE_4(:,idx2.vld)');
                dsX2vld_6 = arrayDatastore(inp2.DATABASE_6(:,idx2.vld)');
                dsX2vld_7 = arrayDatastore(inp2.DATABASE_7(:,idx2.vld)');
                dsT2vld_1 = arrayDatastore(tar2.DATABASE_1(:,idx2.vld)');
                dsT2vld_2 = arrayDatastore(tar2.DATABASE_2(:,idx2.vld)');
                dsT2vld_3 = arrayDatastore(tar2.DATABASE_3(:,idx2.vld)');
                dsVld2= combine(dsX2vld_1,dsX2vld_2,dsX2vld_5,dsX2vld_3,dsX2vld_4,dsX2vld_6,dsX2vld_7,dsT2vld_1,dsT2vld_2,dsT2vld_3);
            end

        elseif index_extra>0 && n_classes>0 && n_fm>0
            NNs{i_}.inp.trn = {inp.DATABASE_1(:,dsg.idx.trn)',inp.DATABASE_2(:,dsg.idx.trn)',inp.DATABASE_5(:,dsg.idx.trn)',inp.DATABASE_3(:,dsg.idx.trn)',inp.DATABASE_4(:,dsg.idx.trn)',inp.DATABASE_6(:,dsg.idx.trn)'};
            NNs{i_}.tar.trn = {tar.DATABASE_1(:,dsg.idx.trn)',tar.DATABASE_2(:,dsg.idx.trn)',tar.DATABASE_3(:,dsg.idx.trn)'};
            NNs{i_}.inp.vld = {inp.DATABASE_1(:,dsg.idx.vld)',inp.DATABASE_2(:,dsg.idx.vld)',inp.DATABASE_5(:,dsg.idx.vld)',inp.DATABASE_3(:,dsg.idx.vld)',inp.DATABASE_4(:,dsg.idx.vld)',inp.DATABASE_6(:,dsg.idx.vld)'};
            NNs{i_}.tar.vld = {tar.DATABASE_1(:,dsg.idx.vld)',tar.DATABASE_2(:,dsg.idx.vld)',tar.DATABASE_3(:,dsg.idx.vld)'};
            NNs{i_}.inp.tst = {inp.DATABASE_1(:,dsg.idx.tst)',inp.DATABASE_2(:,dsg.idx.tst)',inp.DATABASE_5(:,dsg.idx.tst)',inp.DATABASE_3(:,dsg.idx.tst)',inp.DATABASE_4(:,dsg.idx.tst)',inp.DATABASE_6(:,dsg.idx.tst)'};
            NNs{i_}.tar.tst = {tar.DATABASE_1(:,dsg.idx.tst)',tar.DATABASE_2(:,dsg.idx.tst)',tar.DATABASE_3(:,dsg.idx.tst)'};

            dsX1Trn_1 = arrayDatastore(inp.DATABASE_1(:,dsg.idx.trn)');
            dsX1Trn_2 = arrayDatastore(inp.DATABASE_2(:,dsg.idx.trn)');
            dsX1Trn_5 = arrayDatastore(inp.DATABASE_5(:,dsg.idx.trn)');
            dsX1Trn_3 = arrayDatastore(inp.DATABASE_3(:,dsg.idx.trn)');
            dsX1Trn_4 = arrayDatastore(inp.DATABASE_4(:,dsg.idx.trn)');
            dsX1Trn_6 = arrayDatastore(inp.DATABASE_6(:,dsg.idx.trn)');
            dsT1Trn_1 = arrayDatastore(tar.DATABASE_1(:,dsg.idx.trn)');
            dsT1Trn_2 = arrayDatastore(tar.DATABASE_2(:,dsg.idx.trn)');
            dsT1Trn_3 = arrayDatastore(tar.DATABASE_3(:,dsg.idx.trn)');
            dsTrn1 = combine(dsX1Trn_1,dsX1Trn_2,dsX1Trn_5,dsX1Trn_3,dsX1Trn_4,dsX1Trn_6,dsT1Trn_1,dsT1Trn_2,dsT1Trn_3);

            dsX1vld_1 = arrayDatastore(inp.DATABASE_1(:,dsg.idx.vld)');
            dsX1vld_2 = arrayDatastore(inp.DATABASE_2(:,dsg.idx.vld)');
            dsX1vld_5 = arrayDatastore(inp.DATABASE_5(:,dsg.idx.vld)');
            dsX1vld_3 = arrayDatastore(inp.DATABASE_3(:,dsg.idx.vld)');
            dsX1vld_4 = arrayDatastore(inp.DATABASE_4(:,dsg.idx.vld)');
            dsX1vld_6 = arrayDatastore(inp.DATABASE_6(:,dsg.idx.vld)');
            dsT1vld_1 = arrayDatastore(tar.DATABASE_1(:,dsg.idx.vld)');
            dsT1vld_2 = arrayDatastore(tar.DATABASE_2(:,dsg.idx.vld)');
            dsT1vld_3 = arrayDatastore(tar.DATABASE_3(:,dsg.idx.vld)');
            dsVld1= combine(dsX1vld_1,dsX1vld_2,dsX1vld_5,dsX1vld_3,dsX1vld_4,dsX1vld_6,dsT1vld_1,dsT1vld_2,dsT1vld_3);

            if strcmp(TransferLearning,'True')
                NNs{i_}.inp2.trn = {inp2.DATABASE_1(:,idx2.trn)',inp2.DATABASE_2(:,idx2.trn)',inp2.DATABASE_5(:,idx2.trn)',inp2.DATABASE_3(:,idx2.trn)',inp2.DATABASE_4(:,idx2.trn)',inp2.DATABASE_6(:,idx2.trn)'};
                NNs{i_}.tar2.trn = {tar2.DATABASE_1(:,idx2.trn)',tar2.DATABASE_2(:,idx2.trn)',tar2.DATABASE_3(:,idx2.trn)'};
                NNs{i_}.inp2.vld = {inp2.DATABASE_1(:,idx2.vld)',inp2.DATABASE_2(:,idx2.vld)',inp2.DATABASE_5(:,idx2.vld)',inp2.DATABASE_3(:,idx2.vld)',inp2.DATABASE_4(:,idx2.vld)',inp2.DATABASE_6(:,idx2.vld)'};
                NNs{i_}.tar2.vld = {tar2.DATABASE_1(:,idx2.vld)',tar2.DATABASE_2(:,idx2.vld)',tar2.DATABASE_3(:,idx2.vld)'};
                NNs{i_}.inp2.tst = {inp2.DATABASE_1(:,idx2.tst)',inp2.DATABASE_2(:,idx2.tst)',inp2.DATABASE_5(:,idx2.tst)',inp2.DATABASE_3(:,idx2.tst)',inp2.DATABASE_4(:,idx2.tst)',inp2.DATABASE_6(:,idx2.tst)'};
                NNs{i_}.tar2.tst = {tar2.DATABASE_1(:,idx2.tst)',tar2.DATABASE_2(:,idx2.tst)',tar2.DATABASE_3(:,idx2.tst)'};

                dsX2Trn_1 = arrayDatastore(inp2.DATABASE_1(:,idx2.trn)');
                dsX2Trn_2 = arrayDatastore(inp2.DATABASE_2(:,idx2.trn)');
                dsX2Trn_5 = arrayDatastore(inp2.DATABASE_5(:,idx2.trn)');
                dsX2Trn_3 = arrayDatastore(inp2.DATABASE_3(:,idx2.trn)');
                dsX2Trn_4 = arrayDatastore(inp2.DATABASE_4(:,idx2.trn)');
                dsX2Trn_6 = arrayDatastore(inp2.DATABASE_6(:,idx2.trn)');
                dsT2Trn_1 = arrayDatastore(tar2.DATABASE_1(:,idx2.trn)');
                dsT2Trn_2 = arrayDatastore(tar2.DATABASE_2(:,idx2.trn)');
                dsT2Trn_3 = arrayDatastore(tar2.DATABASE_3(:,idx2.trn)');
                dsTrn2 = combine(dsX2Trn_1,dsX2Trn_2,dsX2Trn_5,dsX2Trn_3,dsX2Trn_4,dsX2Trn_6,dsT2Trn_1,dsT2Trn_2,dsT2Trn_3);

                dsX2vld_1 = arrayDatastore(inp2.DATABASE_1(:,idx2.vld)');
                dsX2vld_2 = arrayDatastore(inp2.DATABASE_2(:,idx2.vld)');
                dsX2vld_5 = arrayDatastore(inp2.DATABASE_5(:,idx2.vld)');
                dsX2vld_3 = arrayDatastore(inp2.DATABASE_3(:,idx2.vld)');
                dsX2vld_4 = arrayDatastore(inp2.DATABASE_4(:,idx2.vld)');
                dsX2vld_6 = arrayDatastore(inp2.DATABASE_6(:,idx2.vld)');
                dsT2vld_1 = arrayDatastore(tar2.DATABASE_1(:,idx2.vld)');
                dsT2vld_2 = arrayDatastore(tar2.DATABASE_2(:,idx2.vld)');
                dsT2vld_3 = arrayDatastore(tar2.DATABASE_3(:,idx2.vld)');
                dsVld2= combine(dsX2vld_1,dsX2vld_2,dsX2vld_5,dsX2vld_3,dsX2vld_4,dsX2vld_6,dsT2vld_1,dsT2vld_2,dsT2vld_3);
            end

        elseif index_extra>0 && n_rg>0 && n_fm>0
            NNs{i_}.inp.trn = {inp.DATABASE_1(:,dsg.idx.trn)',inp.DATABASE_2(:,dsg.idx.trn)',inp.DATABASE_5(:,dsg.idx.trn)',inp.DATABASE_3(:,dsg.idx.trn)',inp.DATABASE_6(:,dsg.idx.trn)',inp.DATABASE_7(:,dsg.idx.trn)'};
            NNs{i_}.tar.trn = {tar.DATABASE_1(:,dsg.idx.trn)',tar.DATABASE_2(:,dsg.idx.trn)',tar.DATABASE_3(:,dsg.idx.trn)'};
            NNs{i_}.inp.vld = {inp.DATABASE_1(:,dsg.idx.vld)',inp.DATABASE_2(:,dsg.idx.vld)',inp.DATABASE_5(:,dsg.idx.vld)',inp.DATABASE_3(:,dsg.idx.vld)',inp.DATABASE_6(:,dsg.idx.vld)',inp.DATABASE_7(:,dsg.idx.vld)'};
            NNs{i_}.tar.vld = {tar.DATABASE_1(:,dsg.idx.vld)',tar.DATABASE_2(:,dsg.idx.vld)',tar.DATABASE_3(:,dsg.idx.vld)'};
            NNs{i_}.inp.tst = {inp.DATABASE_1(:,dsg.idx.tst)',inp.DATABASE_2(:,dsg.idx.tst)',inp.DATABASE_5(:,dsg.idx.tst)',inp.DATABASE_3(:,dsg.idx.tst)',inp.DATABASE_6(:,dsg.idx.tst)',inp.DATABASE_7(:,dsg.idx.tst)'};
            NNs{i_}.tar.tst = {tar.DATABASE_1(:,dsg.idx.tst)',tar.DATABASE_2(:,dsg.idx.tst)',tar.DATABASE_3(:,dsg.idx.tst)'};

            dsX1Trn_1 = arrayDatastore(inp.DATABASE_1(:,dsg.idx.trn)');
            dsX1Trn_2 = arrayDatastore(inp.DATABASE_2(:,dsg.idx.trn)');
            dsX1Trn_5 = arrayDatastore(inp.DATABASE_5(:,dsg.idx.trn)');
            dsX1Trn_3 = arrayDatastore(inp.DATABASE_3(:,dsg.idx.trn)');
            dsX1Trn_7 = arrayDatastore(inp.DATABASE_7(:,dsg.idx.trn)');
            dsX1Trn_6 = arrayDatastore(inp.DATABASE_6(:,dsg.idx.trn)');
            dsT1Trn_1 = arrayDatastore(tar.DATABASE_1(:,dsg.idx.trn)');
            dsT1Trn_2 = arrayDatastore(tar.DATABASE_2(:,dsg.idx.trn)');
            dsT1Trn_3 = arrayDatastore(tar.DATABASE_3(:,dsg.idx.trn)');
            dsTrn1 = combine(dsX1Trn_1,dsX1Trn_2,dsX1Trn_5,dsX1Trn_3,dsX1Trn_6,dsX1Trn_7,dsT1Trn_1,dsT1Trn_2,dsT1Trn_3);

            dsX1vld_1 = arrayDatastore(inp.DATABASE_1(:,dsg.idx.vld)');
            dsX1vld_2 = arrayDatastore(inp.DATABASE_2(:,dsg.idx.vld)');
            dsX1vld_5 = arrayDatastore(inp.DATABASE_5(:,dsg.idx.vld)');
            dsX1vld_3 = arrayDatastore(inp.DATABASE_3(:,dsg.idx.vld)');
            dsX1vld_7 = arrayDatastore(inp.DATABASE_7(:,dsg.idx.vld)');
            dsX1vld_6 = arrayDatastore(inp.DATABASE_6(:,dsg.idx.vld)');
            dsT1vld_1 = arrayDatastore(tar.DATABASE_1(:,dsg.idx.vld)');
            dsT1vld_2 = arrayDatastore(tar.DATABASE_2(:,dsg.idx.vld)');
            dsT1vld_3 = arrayDatastore(tar.DATABASE_3(:,dsg.idx.vld)');
            dsVld1= combine(dsX1vld_1,dsX1vld_2,dsX1vld_5,dsX1vld_3,dsX1vld_6,dsX1vld_7,dsT1vld_1,dsT1vld_2,dsT1vld_3);

            if strcmp(TransferLearning,'True')
                NNs{i_}.inp2.trn = {inp2.DATABASE_1(:,idx2.trn)',inp2.DATABASE_2(:,idx2.trn)',inp2.DATABASE_5(:,idx2.trn)',inp2.DATABASE_3(:,idx2.trn)',inp2.DATABASE_6(:,idx2.trn)',inp2.DATABASE_7(:,idx2.trn)'};
                NNs{i_}.tar2.trn = {tar2.DATABASE_1(:,idx2.trn)',tar2.DATABASE_2(:,idx2.trn)',tar2.DATABASE_3(:,idx2.trn)'};
                NNs{i_}.inp2.vld = {inp2.DATABASE_1(:,idx2.vld)',inp2.DATABASE_2(:,idx2.vld)',inp2.DATABASE_5(:,idx2.vld)',inp2.DATABASE_3(:,idx2.vld)',inp2.DATABASE_6(:,idx2.vld)',inp2.DATABASE_7(:,idx2.vld)'};
                NNs{i_}.tar2.vld = {tar2.DATABASE_1(:,idx2.vld)',tar2.DATABASE_2(:,idx2.vld)',tar2.DATABASE_3(:,idx2.vld)'};
                NNs{i_}.inp2.tst = {inp2.DATABASE_1(:,idx2.tst)',inp2.DATABASE_2(:,idx2.tst)',inp2.DATABASE_5(:,idx2.tst)',inp2.DATABASE_3(:,idx2.tst)',inp2.DATABASE_6(:,idx2.tst)',inp2.DATABASE_7(:,idx2.tst)'};
                NNs{i_}.tar2.tst = {tar2.DATABASE_1(:,idx2.tst)',tar2.DATABASE_2(:,idx2.tst)',tar2.DATABASE_3(:,idx2.tst)'};

                dsX2Trn_1 = arrayDatastore(inp2.DATABASE_1(:,idx2.trn)');
                dsX2Trn_2 = arrayDatastore(inp2.DATABASE_2(:,idx2.trn)');
                dsX2Trn_5 = arrayDatastore(inp2.DATABASE_5(:,idx2.trn)');
                dsX2Trn_3 = arrayDatastore(inp2.DATABASE_3(:,idx2.trn)');
                dsX2Trn_7 = arrayDatastore(inp2.DATABASE_7(:,idx2.trn)');
                dsX2Trn_6 = arrayDatastore(inp2.DATABASE_6(:,idx2.trn)');
                dsT2Trn_1 = arrayDatastore(tar2.DATABASE_1(:,idx2.trn)');
                dsT2Trn_2 = arrayDatastore(tar2.DATABASE_2(:,idx2.trn)');
                dsT2Trn_3 = arrayDatastore(tar2.DATABASE_3(:,idx2.trn)');
                dsTrn2 = combine(dsX2Trn_1,dsX2Trn_2,dsX2Trn_5,dsX2Trn_3,dsX2Trn_6,dsX2Trn_7,dsT2Trn_1,dsT2Trn_2,dsT2Trn_3);

                dsX2vld_1 = arrayDatastore(inp2.DATABASE_1(:,idx2.vld)');
                dsX2vld_2 = arrayDatastore(inp2.DATABASE_2(:,idx2.vld)');
                dsX2vld_5 = arrayDatastore(inp2.DATABASE_5(:,idx2.vld)');
                dsX2vld_3 = arrayDatastore(inp2.DATABASE_3(:,idx2.vld)');
                dsX2vld_7 = arrayDatastore(inp2.DATABASE_7(:,idx2.vld)');
                dsX2vld_6 = arrayDatastore(inp2.DATABASE_6(:,idx2.vld)');
                dsT2vld_1 = arrayDatastore(tar2.DATABASE_1(:,idx2.vld)');
                dsT2vld_2 = arrayDatastore(tar2.DATABASE_2(:,idx2.vld)');
                dsT2vld_3 = arrayDatastore(tar2.DATABASE_3(:,idx2.vld)');
                dsVld2= combine(dsX2vld_1,dsX2vld_2,dsX2vld_5,dsX2vld_3,dsX2vld_6,dsX2vld_7,dsT2vld_1,dsT2vld_2,dsT2vld_3);
            end

        elseif index_extra>0 && n_classes>0
            NNs{i_}.inp.trn = {inp.DATABASE_1(:,dsg.idx.trn)',inp.DATABASE_2(:,dsg.idx.trn)',inp.DATABASE_5(:,dsg.idx.trn)',inp.DATABASE_3(:,dsg.idx.trn)',inp.DATABASE_4(:,dsg.idx.trn)'};
            NNs{i_}.tar.trn = {tar.DATABASE_1(:,dsg.idx.trn)',tar.DATABASE_2(:,dsg.idx.trn)',tar.DATABASE_3(:,dsg.idx.trn)'};
            NNs{i_}.inp.vld = {inp.DATABASE_1(:,dsg.idx.vld)',inp.DATABASE_2(:,dsg.idx.vld)',inp.DATABASE_5(:,dsg.idx.vld)',inp.DATABASE_3(:,dsg.idx.vld)',inp.DATABASE_4(:,dsg.idx.vld)'};
            NNs{i_}.tar.vld = {tar.DATABASE_1(:,dsg.idx.vld)',tar.DATABASE_2(:,dsg.idx.vld)',tar.DATABASE_3(:,dsg.idx.vld)'};
            NNs{i_}.inp.tst = {inp.DATABASE_1(:,dsg.idx.tst)',inp.DATABASE_2(:,dsg.idx.tst)',inp.DATABASE_5(:,dsg.idx.tst)',inp.DATABASE_3(:,dsg.idx.tst)',inp.DATABASE_4(:,dsg.idx.tst)'};
            NNs{i_}.tar.tst = {tar.DATABASE_1(:,dsg.idx.tst)',tar.DATABASE_2(:,dsg.idx.tst)',tar.DATABASE_3(:,dsg.idx.tst)'};

            dsX1Trn_1 = arrayDatastore(inp.DATABASE_1(:,dsg.idx.trn)');
            dsX1Trn_2 = arrayDatastore(inp.DATABASE_2(:,dsg.idx.trn)');
            dsX1Trn_5 = arrayDatastore(inp.DATABASE_5(:,dsg.idx.trn)');
            dsX1Trn_3 = arrayDatastore(inp.DATABASE_3(:,dsg.idx.trn)');
            dsX1Trn_4 = arrayDatastore(inp.DATABASE_4(:,dsg.idx.trn)');
            dsT1Trn_1 = arrayDatastore(tar.DATABASE_1(:,dsg.idx.trn)');
            dsT1Trn_2 = arrayDatastore(tar.DATABASE_2(:,dsg.idx.trn)');
            dsT1Trn_3 = arrayDatastore(tar.DATABASE_3(:,dsg.idx.trn)');
            dsTrn1 = combine(dsX1Trn_1,dsX1Trn_2,dsX1Trn_5,dsX1Trn_3,dsX1Trn_4,dsT1Trn_1,dsT1Trn_2,dsT1Trn_3);

            dsX1vld_1 = arrayDatastore(inp.DATABASE_1(:,dsg.idx.vld)');
            dsX1vld_2 = arrayDatastore(inp.DATABASE_2(:,dsg.idx.vld)');
            dsX1vld_5 = arrayDatastore(inp.DATABASE_5(:,dsg.idx.vld)');
            dsX1vld_3 = arrayDatastore(inp.DATABASE_3(:,dsg.idx.vld)');
            dsX1vld_4 = arrayDatastore(inp.DATABASE_4(:,dsg.idx.vld)');
            dsT1vld_1 = arrayDatastore(tar.DATABASE_1(:,dsg.idx.vld)');
            dsT1vld_2 = arrayDatastore(tar.DATABASE_2(:,dsg.idx.vld)');
            dsT1vld_3 = arrayDatastore(tar.DATABASE_3(:,dsg.idx.vld)');
            dsVld1= combine(dsX1vld_1,dsX1vld_2,dsX1vld_5,dsX1vld_3,dsX1vld_4,dsT1vld_1,dsT1vld_2,dsT1vld_3);

            if strcmp(TransferLearning,'True')
                NNs{i_}.inp2.trn = {inp2.DATABASE_1(:,idx2.trn)',inp2.DATABASE_2(:,idx2.trn)',inp2.DATABASE_5(:,idx2.trn)',inp2.DATABASE_3(:,idx2.trn)',inp2.DATABASE_4(:,idx2.trn)'};
                NNs{i_}.tar2.trn = {tar2.DATABASE_1(:,idx2.trn)',tar2.DATABASE_2(:,idx2.trn)',tar2.DATABASE_3(:,idx2.trn)'};
                NNs{i_}.inp2.vld = {inp2.DATABASE_1(:,idx2.vld)',inp2.DATABASE_2(:,idx2.vld)',inp2.DATABASE_5(:,idx2.vld)',inp2.DATABASE_3(:,idx2.vld)',inp2.DATABASE_4(:,idx2.vld)'};
                NNs{i_}.tar2.vld = {tar2.DATABASE_1(:,idx2.vld)',tar2.DATABASE_2(:,idx2.vld)',tar2.DATABASE_3(:,idx2.vld)'};
                NNs{i_}.inp2.tst = {inp2.DATABASE_1(:,idx2.tst)',inp2.DATABASE_2(:,idx2.tst)',inp2.DATABASE_5(:,idx2.tst)',inp2.DATABASE_3(:,idx2.tst)',inp2.DATABASE_4(:,idx2.tst)'};
                NNs{i_}.tar2.tst = {tar2.DATABASE_1(:,idx2.tst)',tar2.DATABASE_2(:,idx2.tst)',tar2.DATABASE_3(:,idx2.tst)'};

                dsX2Trn_1 = arrayDatastore(inp2.DATABASE_1(:,idx2.trn)');
                dsX2Trn_2 = arrayDatastore(inp2.DATABASE_2(:,idx2.trn)');
                dsX2Trn_5 = arrayDatastore(inp2.DATABASE_5(:,idx2.trn)');
                dsX2Trn_3 = arrayDatastore(inp2.DATABASE_3(:,idx2.trn)');
                dsX2Trn_4 = arrayDatastore(inp2.DATABASE_4(:,idx2.trn)');
                dsT2Trn_1 = arrayDatastore(tar2.DATABASE_1(:,idx2.trn)');
                dsT2Trn_2 = arrayDatastore(tar2.DATABASE_2(:,idx2.trn)');
                dsT2Trn_3 = arrayDatastore(tar2.DATABASE_3(:,idx2.trn)');
                dsTrn2 = combine(dsX2Trn_1,dsX2Trn_2,dsX2Trn_5,dsX2Trn_3,dsX2Trn_4,dsT2Trn_1,dsT2Trn_2,dsT2Trn_3);

                dsX2vld_1 = arrayDatastore(inp2.DATABASE_1(:,idx2.vld)');
                dsX2vld_2 = arrayDatastore(inp2.DATABASE_2(:,idx2.vld)');
                dsX2vld_5 = arrayDatastore(inp2.DATABASE_5(:,idx2.vld)');
                dsX2vld_3 = arrayDatastore(inp2.DATABASE_3(:,idx2.vld)');
                dsX2vld_4 = arrayDatastore(inp2.DATABASE_4(:,idx2.vld)');
                dsT2vld_1 = arrayDatastore(tar2.DATABASE_1(:,idx2.vld)');
                dsT2vld_2 = arrayDatastore(tar2.DATABASE_2(:,idx2.vld)');
                dsT2vld_3 = arrayDatastore(tar2.DATABASE_3(:,idx2.vld)');
                dsVld2= combine(dsX2vld_1,dsX2vld_2,dsX2vld_5,dsX2vld_3,dsX2vld_4,dsT2vld_1,dsT2vld_2,dsT2vld_3);
            end

        elseif (index_extra>0) && (n_classes==0)
            NNs{i_}.inp.trn = {inp.DATABASE_1(:,dsg.idx.trn)',inp.DATABASE_2(:,dsg.idx.trn)',inp.DATABASE_5(:,dsg.idx.trn)',inp.DATABASE_3(:,dsg.idx.trn)'};
            NNs{i_}.tar.trn = {tar.DATABASE_1(:,dsg.idx.trn)',tar.DATABASE_2(:,dsg.idx.trn)',tar.DATABASE_3(:,dsg.idx.trn)'};
            NNs{i_}.inp.vld = {inp.DATABASE_1(:,dsg.idx.vld)',inp.DATABASE_2(:,dsg.idx.vld)',inp.DATABASE_5(:,dsg.idx.vld)',inp.DATABASE_3(:,dsg.idx.vld)'};
            NNs{i_}.tar.vld = {tar.DATABASE_1(:,dsg.idx.vld)',tar.DATABASE_2(:,dsg.idx.vld)',tar.DATABASE_3(:,dsg.idx.vld)'};
            NNs{i_}.inp.tst = {inp.DATABASE_1(:,dsg.idx.tst)',inp.DATABASE_2(:,dsg.idx.tst)',inp.DATABASE_5(:,dsg.idx.tst)',inp.DATABASE_3(:,dsg.idx.tst)'};
            NNs{i_}.tar.tst = {tar.DATABASE_1(:,dsg.idx.tst)',tar.DATABASE_2(:,dsg.idx.tst)',tar.DATABASE_3(:,dsg.idx.tst)'};

            dsX1Trn_1 = arrayDatastore(inp.DATABASE_1(:,dsg.idx.trn)');
            dsX1Trn_2 = arrayDatastore(inp.DATABASE_2(:,dsg.idx.trn)');
            dsX1Trn_5 = arrayDatastore(inp.DATABASE_5(:,dsg.idx.trn)');
            dsX1Trn_3 = arrayDatastore(inp.DATABASE_3(:,dsg.idx.trn)');
            dsT1Trn_1 = arrayDatastore(tar.DATABASE_1(:,dsg.idx.trn)');
            dsT1Trn_2 = arrayDatastore(tar.DATABASE_2(:,dsg.idx.trn)');
            dsT1Trn_3 = arrayDatastore(tar.DATABASE_3(:,dsg.idx.trn)');
            dsTrn1 = combine(dsX1Trn_1,dsX1Trn_2,dsX1Trn_5,dsX1Trn_3,dsT1Trn_1,dsT1Trn_2,dsT1Trn_3);

            dsX1vld_1 = arrayDatastore(inp.DATABASE_1(:,dsg.idx.vld)');
            dsX1vld_2 = arrayDatastore(inp.DATABASE_2(:,dsg.idx.vld)');
            dsX1vld_5 = arrayDatastore(inp.DATABASE_5(:,dsg.idx.vld)');
            dsX1vld_3 = arrayDatastore(inp.DATABASE_3(:,dsg.idx.vld)');
            dsT1vld_1 = arrayDatastore(tar.DATABASE_1(:,dsg.idx.vld)');
            dsT1vld_2 = arrayDatastore(tar.DATABASE_2(:,dsg.idx.vld)');
            dsT1vld_3 = arrayDatastore(tar.DATABASE_3(:,dsg.idx.vld)');
            dsVld1= combine(dsX1vld_1,dsX1vld_2,dsX1vld_5,dsX1vld_3,dsT1vld_1,dsT1vld_2,dsT1vld_3);

            if strcmp(TransferLearning,'True')
                NNs{i_}.inp2.trn = {inp2.DATABASE_1(:,idx2.trn)',inp2.DATABASE_2(:,idx2.trn)',inp2.DATABASE_5(:,idx2.trn)',inp2.DATABASE_3(:,idx2.trn)'};
                NNs{i_}.tar2.trn = {tar2.DATABASE_1(:,idx2.trn)',tar2.DATABASE_2(:,idx2.trn)',tar2.DATABASE_3(:,idx2.trn)'};
                NNs{i_}.inp2.vld = {inp2.DATABASE_1(:,idx2.vld)',inp2.DATABASE_2(:,idx2.vld)',inp2.DATABASE_5(:,idx2.vld)',inp2.DATABASE_3(:,idx2.vld)'};
                NNs{i_}.tar2.vld = {tar2.DATABASE_1(:,idx2.vld)',tar2.DATABASE_2(:,idx2.vld)',tar2.DATABASE_3(:,idx2.vld)'};
                NNs{i_}.inp2.tst = {inp2.DATABASE_1(:,idx2.tst)',inp2.DATABASE_2(:,idx2.tst)',inp2.DATABASE_5(:,idx2.tst)',inp2.DATABASE_3(:,idx2.tst)'};
                NNs{i_}.tar2.tst = {tar2.DATABASE_1(:,idx2.tst)',tar2.DATABASE_2(:,idx2.tst)',tar2.DATABASE_3(:,idx2.tst)'};

                dsX2Trn_1 = arrayDatastore(inp2.DATABASE_1(:,idx2.trn)');
                dsX2Trn_2 = arrayDatastore(inp2.DATABASE_2(:,idx2.trn)');
                dsX2Trn_5 = arrayDatastore(inp2.DATABASE_5(:,idx2.trn)');
                dsX2Trn_3 = arrayDatastore(inp2.DATABASE_3(:,idx2.trn)');
                dsT2Trn_1 = arrayDatastore(tar2.DATABASE_1(:,idx2.trn)');
                dsT2Trn_2 = arrayDatastore(tar2.DATABASE_2(:,idx2.trn)');
                dsT2Trn_3 = arrayDatastore(tar2.DATABASE_3(:,idx2.trn)');
                dsTrn2 = combine(dsX2Trn_1,dsX2Trn_2,dsX2Trn_5,dsX2Trn_3,dsT2Trn_1,dsT2Trn_2,dsT2Trn_3);

                dsX2vld_1 = arrayDatastore(inp2.DATABASE_1(:,idx2.vld)');
                dsX2vld_2 = arrayDatastore(inp2.DATABASE_2(:,idx2.vld)');
                dsX2vld_5 = arrayDatastore(inp2.DATABASE_5(:,idx2.vld)');
                dsX2vld_3 = arrayDatastore(inp2.DATABASE_3(:,idx2.vld)');
                dsT2vld_1 = arrayDatastore(tar2.DATABASE_1(:,idx2.vld)');
                dsT2vld_2 = arrayDatastore(tar2.DATABASE_2(:,idx2.vld)');
                dsT2vld_3 = arrayDatastore(tar2.DATABASE_3(:,idx2.vld)');
                dsVld2= combine(dsX2vld_1,dsX2vld_2,dsX2vld_5,dsX2vld_3,dsT2vld_1,dsT2vld_2,dsT2vld_3);
            end

        else
            NNs{i_}.inp.trn = {inp.DATABASE_1(:,dsg.idx.trn)',inp.DATABASE_2(:,dsg.idx.trn)',inp.DATABASE_5(:,dsg.idx.trn)'};
            NNs{i_}.tar.trn = {tar.DATABASE_1(:,dsg.idx.trn)',tar.DATABASE_2(:,dsg.idx.trn)',tar.DATABASE_3(:,dsg.idx.trn)'};
            NNs{i_}.inp.vld = {inp.DATABASE_1(:,dsg.idx.vld)',inp.DATABASE_2(:,dsg.idx.vld)',inp.DATABASE_5(:,dsg.idx.vld)'};
            NNs{i_}.tar.vld = {tar.DATABASE_1(:,dsg.idx.vld)',tar.DATABASE_2(:,dsg.idx.vld)',tar.DATABASE_3(:,dsg.idx.vld)'};
            NNs{i_}.inp.tst = {inp.DATABASE_1(:,dsg.idx.tst)',inp.DATABASE_2(:,dsg.idx.tst)',inp.DATABASE_5(:,dsg.idx.tst)'};
            NNs{i_}.tar.tst = {tar.DATABASE_1(:,dsg.idx.tst)',tar.DATABASE_2(:,dsg.idx.tst)',tar.DATABASE_3(:,dsg.idx.tst)'};

            dsX1Trn_1 = arrayDatastore(inp.DATABASE_1(:,dsg.idx.trn)');
            dsX1Trn_2 = arrayDatastore(inp.DATABASE_2(:,dsg.idx.trn)');
            dsX1Trn_5 = arrayDatastore(inp.DATABASE_5(:,dsg.idx.trn)');
            dsT1Trn_1 = arrayDatastore(tar.DATABASE_1(:,dsg.idx.trn)');
            dsT1Trn_2 = arrayDatastore(tar.DATABASE_2(:,dsg.idx.trn)');
            dsT1Trn_3 = arrayDatastore(tar.DATABASE_3(:,dsg.idx.trn)');
            dsTrn1 = combine(dsX1Trn_1,dsX1Trn_2,dsX1Trn_5,dsT1Trn_1,dsT1Trn_2,dsT1Trn_3);

            dsX1vld_1 = arrayDatastore(inp.DATABASE_1(:,dsg.idx.vld)');
            dsX1vld_2 = arrayDatastore(inp.DATABASE_2(:,dsg.idx.vld)');
            dsX1vld_5 = arrayDatastore(inp.DATABASE_5(:,dsg.idx.vld)');
            dsT1vld_1 = arrayDatastore(tar.DATABASE_1(:,dsg.idx.vld)');
            dsT1vld_2 = arrayDatastore(tar.DATABASE_2(:,dsg.idx.vld)');
            dsT1vld_3 = arrayDatastore(tar.DATABASE_3(:,dsg.idx.vld)');
            dsVld1= combine(dsX1vld_1,dsX1vld_2,dsX1vld_5,dsT1vld_1,dsT1vld_2,dsT1vld_3);

            if strcmp(TransferLearning,'True')
                NNs{i_}.inp2.trn = {inp2.DATABASE_1(:,idx2.trn)',inp2.DATABASE_2(:,idx2.trn)',inp2.DATABASE_5(:,idx2.trn)'};
                NNs{i_}.tar2.trn = {tar2.DATABASE_1(:,idx2.trn)',tar2.DATABASE_2(:,idx2.trn)',tar2.DATABASE_3(:,idx2.trn)'};
                NNs{i_}.inp2.vld = {inp2.DATABASE_1(:,idx2.vld)',inp2.DATABASE_2(:,idx2.vld)',inp2.DATABASE_5(:,idx2.vld)'};
                NNs{i_}.tar2.vld = {tar2.DATABASE_1(:,idx2.vld)',tar2.DATABASE_2(:,idx2.vld)',tar2.DATABASE_3(:,idx2.vld)'};
                NNs{i_}.inp2.tst = {inp2.DATABASE_1(:,idx2.tst)',inp2.DATABASE_2(:,idx2.tst)',inp2.DATABASE_5(:,idx2.tst)'};
                NNs{i_}.tar2.tst = {tar2.DATABASE_1(:,idx2.tst)',tar2.DATABASE_2(:,idx2.tst)',tar2.DATABASE_3(:,idx2.tst)'};

                dsX2Trn_1 = arrayDatastore(inp2.DATABASE_1(:,idx2.trn)');
                dsX2Trn_2 = arrayDatastore(inp2.DATABASE_2(:,idx2.trn)');
                dsX2Trn_5 = arrayDatastore(inp2.DATABASE_5(:,idx2.trn)');
                dsT2Trn_1 = arrayDatastore(tar2.DATABASE_1(:,idx2.trn)');
                dsT2Trn_2 = arrayDatastore(tar2.DATABASE_2(:,idx2.trn)');
                dsT2Trn_3 = arrayDatastore(tar2.DATABASE_3(:,idx2.trn)');
                dsTrn2 = combine(dsX2Trn_1,dsX2Trn_2,dsX2Trn_5,dsT2Trn_1,dsT2Trn_2,dsT2Trn_3);

                dsX2vld_1 = arrayDatastore(inp2.DATABASE_1(:,idx2.vld)');
                dsX2vld_2 = arrayDatastore(inp2.DATABASE_2(:,idx2.vld)');
                dsX2vld_5 = arrayDatastore(inp2.DATABASE_5(:,idx2.vld)');
                dsT2vld_1 = arrayDatastore(tar2.DATABASE_1(:,idx2.vld)');
                dsT2vld_2 = arrayDatastore(tar2.DATABASE_2(:,idx2.vld)');
                dsT2vld_3 = arrayDatastore(tar2.DATABASE_3(:,idx2.vld)');
                dsVld2= combine(dsX2vld_1,dsX2vld_2,dsX2vld_5,dsT2vld_1,dsT2vld_2,dsT2vld_3);
            end
        end
    else %only one component

        if index_extra>0 && n_classes>0
            NNs{i_}.inp.trn = {inp.DATABASE_1(:,dsg.idx.trn)',inp.DATABASE_3(:,dsg.idx.trn)',inp.DATABASE_4(:,dsg.idx.trn)'};
            NNs{i_}.tar.trn = {tar.DATABASE_1(:,dsg.idx.trn)'};
            NNs{i_}.inp.vld = {inp.DATABASE_1(:,dsg.idx.vld)',inp.DATABASE_3(:,dsg.idx.vld)',inp.DATABASE_4(:,dsg.idx.vld)'};
            NNs{i_}.tar.vld = {tar.DATABASE_1(:,dsg.idx.vld)'};
            NNs{i_}.inp.tst = {inp.DATABASE_1(:,dsg.idx.tst)',inp.DATABASE_3(:,dsg.idx.tst)',inp.DATABASE_4(:,dsg.idx.tst)'};
            NNs{i_}.tar.tst = {tar.DATABASE_1(:,dsg.idx.tst)'};

            dsX1Trn_1 = arrayDatastore(inp.DATABASE_1(:,dsg.idx.trn)');
            dsX1Trn_3 = arrayDatastore(inp.DATABASE_3(:,dsg.idx.trn)');
            dsX1Trn_4 = arrayDatastore(inp.DATABASE_4(:,dsg.idx.trn)');
            dsT1Trn_1 = arrayDatastore(tar.DATABASE_1(:,dsg.idx.trn)');
            dsTrn1 = combine(dsX1Trn_1,dsX1Trn_3,dsX1Trn_4,dsT1Trn_1);

            dsX1vld_1 = arrayDatastore(inp.DATABASE_1(:,dsg.idx.vld)');
            dsX1vld_3 = arrayDatastore(inp.DATABASE_3(:,dsg.idx.vld)');
            dsX1vld_4 = arrayDatastore(inp.DATABASE_4(:,dsg.idx.vld)');
            dsT1vld_1 = arrayDatastore(tar.DATABASE_1(:,dsg.idx.vld)');
            dsVld1= combine(dsX1vld_1,dsX1vld_3,dsX1vld_4,dsT1vld_1);

            if strcmp(TransferLearning,'True')
                NNs{i_}.inp2.trn = {inp2.DATABASE_1(:,idx2.trn)',inp2.DATABASE_3(:,idx2.trn)',inp2.DATABASE_4(:,idx2.trn)'};
                NNs{i_}.tar2.trn = {tar2.DATABASE_1(:,idx2.trn)'};
                NNs{i_}.inp2.vld = {inp2.DATABASE_1(:,idx2.vld)',inp2.DATABASE_3(:,idx2.vld)',inp2.DATABASE_4(:,idx2.vld)'};
                NNs{i_}.tar2.vld = {tar2.DATABASE_1(:,idx2.vld)'};
                NNs{i_}.inp2.tst = {inp2.DATABASE_1(:,idx2.tst)',inp2.DATABASE_3(:,idx2.tst)',inp2.DATABASE_4(:,idx2.tst)'};
                NNs{i_}.tar2.tst = {tar2.DATABASE_1(:,idx2.tst)'};

                dsX2Trn_1 = arrayDatastore(inp2.DATABASE_1(:,idx2.trn)');
                dsX2Trn_3 = arrayDatastore(inp2.DATABASE_3(:,idx2.trn)');
                dsX2Trn_4 = arrayDatastore(inp2.DATABASE_4(:,idx2.trn)');
                dsT2Trn_1 = arrayDatastore(tar2.DATABASE_1(:,idx2.trn)');
                dsTrn2 = combine(dsX2Trn_1,dsX2Trn_3,dsX2Trn_4,dsT2Trn_1);

                dsX2vld_1 = arrayDatastore(inp2.DATABASE_1(:,idx2.vld)');
                dsX2vld_3 = arrayDatastore(inp2.DATABASE_3(:,idx2.vld)');
                dsX2vld_4 = arrayDatastore(inp2.DATABASE_4(:,idx2.vld)');
                dsT2vld_1 = arrayDatastore(tar2.DATABASE_1(:,idx2.vld)');
                dsVld2= combine(dsX2vld_1,dsX2vld_3,dsX2vld_4,dsT2vld_1);
            end

        elseif (index_extra>0) && (n_classes==0)
            NNs{i_}.inp.trn = {inp.DATABASE_1(:,dsg.idx.trn)',inp.DATABASE_3(:,dsg.idx.trn)'};
            NNs{i_}.tar.trn = {tar.DATABASE_1(:,dsg.idx.trn)'};
            NNs{i_}.inp.vld = {inp.DATABASE_1(:,dsg.idx.vld)',inp.DATABASE_3(:,dsg.idx.vld)'};
            NNs{i_}.tar.vld = {tar.DATABASE_1(:,dsg.idx.vld)'};
            NNs{i_}.inp.tst = {inp.DATABASE_1(:,dsg.idx.tst)',inp.DATABASE_3(:,dsg.idx.tst)'};
            NNs{i_}.tar.tst = {tar.DATABASE_1(:,dsg.idx.tst)'};

            dsX1Trn_1 = arrayDatastore(inp.DATABASE_1(:,dsg.idx.trn)');
            dsX1Trn_3 = arrayDatastore(inp.DATABASE_3(:,dsg.idx.trn)');
            dsT1Trn_1 = arrayDatastore(tar.DATABASE_1(:,dsg.idx.trn)');
            dsTrn1 = combine(dsX1Trn_1,dsX1Trn_3,dsT1Trn_1);

            dsX1vld_1 = arrayDatastore(inp.DATABASE_1(:,dsg.idx.vld)');
            dsX1vld_3 = arrayDatastore(inp.DATABASE_3(:,dsg.idx.vld)');
            dsT1vld_1 = arrayDatastore(tar.DATABASE_1(:,dsg.idx.vld)');
            dsVld1= combine(dsX1vld_1,dsX1vld_3,dsT1vld_1);

            if strcmp(TransferLearning,'True')
                NNs{i_}.inp2.trn = {inp2.DATABASE_1(:,idx2.trn)',inp2.DATABASE_3(:,idx2.trn)'};
                NNs{i_}.tar2.trn = {tar2.DATABASE_1(:,idx2.trn)'};
                NNs{i_}.inp2.vld = {inp2.DATABASE_1(:,idx2.vld)',inp2.DATABASE_3(:,idx2.vld)'};
                NNs{i_}.tar2.vld = {tar2.DATABASE_1(:,idx2.vld)'};
                NNs{i_}.inp2.tst = {inp2.DATABASE_1(:,idx2.tst)',inp2.DATABASE_3(:,idx2.tst)'};
                NNs{i_}.tar2.tst = {tar2.DATABASE_1(:,idx2.tst)'};

                dsX2Trn_1 = arrayDatastore(inp2.DATABASE_1(:,idx2.trn)');
                dsX2Trn_3 = arrayDatastore(inp2.DATABASE_3(:,idx2.trn)');
                dsT2Trn_1 = arrayDatastore(tar2.DATABASE_1(:,idx2.trn)');
                dsTrn2 = combine(dsX2Trn_1,dsX2Trn_3,dsT2Trn_1);

                dsX2vld_1 = arrayDatastore(inp2.DATABASE_1(:,idx2.vld)');
                dsX2vld_3 = arrayDatastore(inp2.DATABASE_3(:,idx2.vld)');
                dsT2vld_1 = arrayDatastore(tar2.DATABASE_1(:,idx2.vld)');
                dsVld2= combine(dsX2vld_1,dsX2vld_3,dsT2vld_1);
            end

        else
            NNs{i_}.inp.trn = {inp.DATABASE_1(:,dsg.idx.trn)'};
            NNs{i_}.tar.trn = {tar.DATABASE_1(:,dsg.idx.trn)'};
            NNs{i_}.inp.vld = {inp.DATABASE_1(:,dsg.idx.vld)'};
            NNs{i_}.tar.vld = {tar.DATABASE_1(:,dsg.idx.vld)'};
            NNs{i_}.inp.tst = {inp.DATABASE_1(:,dsg.idx.tst)'};
            NNs{i_}.tar.tst = {tar.DATABASE_1(:,dsg.idx.tst)'};

            dsX1Trn_1 = arrayDatastore(inp.DATABASE_1(:,dsg.idx.trn)');
            dsT1Trn_1 = arrayDatastore(tar.DATABASE_1(:,dsg.idx.trn)');
            dsTrn1 = combine(dsX1Trn_1,dsT1Trn_1);

            dsX2Trn_1 = arrayDatastore(inp2.DATABASE_1(:,idx2.trn)');
            dsT2Trn_1 = arrayDatastore(tar2.DATABASE_1(:,idx2.trn)');
            dsTrn2 = combine(dsX2Trn_1,dsT2Trn_1);

            if strcmp(TransferLearning,'True')
                NNs{i_}.inp2.trn = {inp2.DATABASE_1(:,idx2.trn)'};
                NNs{i_}.tar2.trn = {tar2.DATABASE_1(:,idx2.trn)'};
                NNs{i_}.inp2.vld = {inp2.DATABASE_1(:,idx2.vld)'};
                NNs{i_}.tar2.vld = {tar2.DATABASE_1(:,idx2.vld)'};
                NNs{i_}.inp2.tst = {inp2.DATABASE_1(:,idx2.tst)'};
                NNs{i_}.tar2.tst = {tar2.DATABASE_1(:,idx2.tst)'};

                dsX1vld_1 = arrayDatastore(inp.DATABASE_1(:,dsg.idx.vld)');
                dsT1vld_1 = arrayDatastore(tar.DATABASE_1(:,dsg.idx.vld)');
                dsVld1= combine(dsX1vld_1,dsT1vld_1);

                dsX2vld_1 = arrayDatastore(inp2.DATABASE_1(:,idx2.vld)');
                dsT2vld_1 = arrayDatastore(tar2.DATABASE_1(:,idx2.vld)');
                dsVld2= combine(dsX2vld_1,dsT2vld_1);

            end
        end
    end


    %% TRAINING ANN
    fprintf('TRAINING...\n');

    % analyzeNetwork(layers)

    % ----------Defining composite loss function-----------------
    delta = 0.8;  % in ln units;
    w=ones(length(tar.vTn),1);w(1)=2; % w(2:4)=0.5;

    whuber = @(Y,T,w,delta) sum( w .* mean( ...
        0.5*(abs(Y-T)<=delta).*(Y-T).^2 + ...
        (abs(Y-T)>delta).*delta.*(abs(Y-T)-0.5*delta), ...
        2) ) ./ sum(w);

    % Bias penalty: penalize nonzero mean residual per period (weighted)
    wBias = ones(size(w));
    wBias(2:5) = 4;  wBias(1)=2;   % e.g., enforce bias more strongly at first periods
    wBias = wBias/mean(wBias);
    wbias = @(Y,T,wBias) sum( wBias .* (mean(Y-T,2)).^2 ) / sum(wBias);

    %Total loss
    alpha = 0.08;
    loss1 = @(Y,T,w,delta) whuber(Y,T,w,delta) + alpha*wbias(Y,T,wBias);

    if strcmp(ann.cp,'h12v')
        lossFcn = @(Y1,Y2,Y3,T1,T2,T3) ( ...
            0.375*loss1(Y1,T1,w,delta) + 0.375*loss1(Y2,T2,w,delta) + 0.25*loss1(Y3,T3,w,delta));
    else
        lossFcn = @(Y1,T1) 1*mse(Y1,T1) ;
    end

    %-------------------- Training------------------------
    miniBatchSize = 64;
    numTrain = length(dsg.idx.trn);
    valFreq  = ceil(numTrain/miniBatchSize);

    options = trainingOptions("adam", ...
        "InitialLearnRate", 5e-4, ...
        "LearnRateSchedule","piecewise", ...
        "LearnRateDropPeriod", 30, ...
        "LearnRateDropFactor", 0.5, ...
        "MaxEpochs", 115, ...
        "MiniBatchSize", miniBatchSize, ...
        "Shuffle", "every-epoch", ...
        "L2Regularization", 5e-4, ...
        "GradientThresholdMethod","l2norm", ...
        "GradientThreshold", 1, ...
        "ValidationData", dsVld1, ...
        "ValidationFrequency", valFreq, ...
        "ValidationPatience", 6, ...
        "OutputNetwork","best-validation-loss", ...
        ... %"Plots","training-progress", ...
        "Verbose", false);
    [NNs{i_}.net,NNs{i_}.trs] = trainnet(dsTrn1,layers,lossFcn,options);
    old_net=NNs{i_}.net;

    if strcmp(TransferLearning,'True')
        %-------- LOSS FUNCTION FOR TL ADDING SMOOTHNESS FACTOR ----------
        smooth2 = @(Y) mean( (Y(3:end,:) - 2*Y(2:end-1,:) + Y(1:end-2,:)).^2, "all");

        lambdaS = 1;  %  smoothness factor
        loss1_smooth = @(Y,T,w,delta) loss1(Y,T,w,delta) + lambdaS*smooth2(Y);

        if strcmp(ann.cp,'h12v')
            lossFcn = @(Y1,Y2,Y3,T1,T2,T3) ( ...
                0.375*loss1_smooth(Y1,T1,w,delta) + ...
                0.375*loss1_smooth(Y2,T2,w,delta) + ...
                0.25 *loss1_smooth(Y3,T3,w,delta));
        else
            lossFcn = @(Y1,T1) 1*loss1_smooth(Y1,T1,w,delta) ;
        end


        %-------------- TRANSFER LEARNING ---------------------------
        % Define the layers to freeze for transfer learning
        % layerName = ["fc_shared1","fc_shared2","output1","output2","output3"];
        %
        % layers = freezeNetwork(NNs{i_}.net,LayerNamesToIgnore=layerName);
        %
        % miniBatchSize = 20;
        % numTrain = length(idx2.trn);
        % valFreq  = ceil(numTrain/miniBatchSize);
        %
        % options2 = trainingOptions("adam", ...
        %     "InitialLearnRate", 5e-4, ...
        %     "LearnRateSchedule","piecewise", ...
        %     "LearnRateDropPeriod", 30, ...
        %     "LearnRateDropFactor", 0.5, ...
        %     "MaxEpochs", 100, ...
        %     "MiniBatchSize", miniBatchSize, ...
        %     "Shuffle", "every-epoch", ...
        %     "L2Regularization", 1e-4, ...
        %     "GradientThresholdMethod","l2norm", ...
        %     "GradientThreshold", 1, ...
        %     "ValidationData", dsVld2, ...
        %     "ValidationFrequency", valFreq, ...
        %     "ValidationPatience", 5, ...
        %     "OutputNetwork","best-validation-loss", ...
        %     ..."Plots","training-progress", ...
        %     "Verbose", false);
        %
        % [NNs{i_}.net,NNs{i_}.trs] = trainnet(dsTrn2,layers,lossFcn,options2);

        % Matlab's trainnet doesn't accept LS-2P penalty so we use custom training loop:

        % ---------------- Custom TL training with LS-2P penalty----------------
        layerName = ["input4","fc_shared1","fc_shared2","output1","output2","output3"]; % layers to update
        miniBatchSize = 20; % Should be defined according to the size of the TL dataset
        maxEpochs     = 150;
        learnRate     = 5e-4;
        lambdaSP      = 20;
        gradThresh    = 1;
        patience      = 8;

        Training_TL;
    end

    %% TEST/VALIDATE ANN PERFORMANCE
    NNs{i_} = train_ann_valid(NNs{i_},TransferLearning,index_extra,n_classes,n_fm,n_rg,ann.cp);
    prf.trn(i_,1) = double(NNs{i_}.prf.trn);
    prf.vld(i_,1) = double(NNs{i_}.prf.vld);
    prf.tst(i_,1) = double(NNs{i_}.prf.tst);
    prf.r(i_,1) = double(NNs{i_}.prf.r);
    prf.mae(i_,1) = double(NNs{i_}.prf.mae);
end

%% COMPUTE BEST PERFORMANCE
NNs = trann_train_best_performance(NNs,prf,dsg,wd,dbn_name,verNet);

%% RESIDUALS PLOT
if strcmp(TransferLearning,'True')
    plot_residuals(NNs,inp2,tar2,wd,dsg,dbn_name,verNet,ann.cp,TransferLearning,...
        add_distance,add_m,add_lndistance,separate_classes,add_vs30,add_fm,separate_regions,add_depth,ann);
else
    plot_residuals(NNs,inp,tar,wd,dsg,dbn_name,verNet,ann.cp,TransferLearning,...
        add_distance,add_m,add_lndistance,separate_classes,add_vs30,add_fm,separate_regions,add_depth,ann);
end

%% OUTPUT
saving_net;

return
end
