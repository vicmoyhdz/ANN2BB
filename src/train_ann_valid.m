function [varargout] = train_ann_valid(varargin)
%% *SET-UP*
ann = varargin{1};
TransferLearning = varargin{2};
index_extra = varargin{3};
n_classes = varargin{4};
n_fm = varargin{5};
n_rg = varargin{6};
component = varargin{7};

%% *ANN VALIDATION*
if strcmp(component,'h12v')
    if index_extra>0 && n_classes>0 && n_fm>0 && n_rg>0
        [ann.out_trn.trn{1,1},ann.out_trn.trn{1,2},ann.out_trn.trn{1,3}] = predict(ann.net,ann.inp.trn{1,1},ann.inp.trn{1,2},ann.inp.trn{1,3},ann.inp.trn{1,4},ann.inp.trn{1,5},ann.inp.trn{1,6},ann.inp.trn{1,7});
        [ann.out_trn.vld{1,1},ann.out_trn.vld{1,2},ann.out_trn.vld{1,3}] = predict(ann.net,ann.inp.vld{1,1},ann.inp.vld{1,2},ann.inp.vld{1,3},ann.inp.vld{1,4},ann.inp.vld{1,5},ann.inp.vld{1,6},ann.inp.vld{1,7});
        [ann.out_trn.tst{1,1},ann.out_trn.tst{1,2},ann.out_trn.tst{1,3}] = predict(ann.net,ann.inp.tst{1,1},ann.inp.tst{1,2},ann.inp.tst{1,3},ann.inp.tst{1,4},ann.inp.tst{1,5},ann.inp.tst{1,6},ann.inp.tst{1,7});
    elseif index_extra>0 && n_classes>0 && n_fm>0
        [ann.out_trn.trn{1,1},ann.out_trn.trn{1,2},ann.out_trn.trn{1,3}] = predict(ann.net,ann.inp.trn{1,1},ann.inp.trn{1,2},ann.inp.trn{1,3},ann.inp.trn{1,4},ann.inp.trn{1,5},ann.inp.trn{1,6});
        [ann.out_trn.vld{1,1},ann.out_trn.vld{1,2},ann.out_trn.vld{1,3}] = predict(ann.net,ann.inp.vld{1,1},ann.inp.vld{1,2},ann.inp.vld{1,3},ann.inp.vld{1,4},ann.inp.vld{1,5},ann.inp.vld{1,6});
        [ann.out_trn.tst{1,1},ann.out_trn.tst{1,2},ann.out_trn.tst{1,3}] = predict(ann.net,ann.inp.tst{1,1},ann.inp.tst{1,2},ann.inp.tst{1,3},ann.inp.tst{1,4},ann.inp.tst{1,5},ann.inp.tst{1,6});
    elseif index_extra>0 && n_rg>0 && n_fm>0
        [ann.out_trn.trn{1,1},ann.out_trn.trn{1,2},ann.out_trn.trn{1,3}] = predict(ann.net,ann.inp.trn{1,1},ann.inp.trn{1,2},ann.inp.trn{1,3},ann.inp.trn{1,4},ann.inp.trn{1,5},ann.inp.trn{1,6});
        [ann.out_trn.vld{1,1},ann.out_trn.vld{1,2},ann.out_trn.vld{1,3}] = predict(ann.net,ann.inp.vld{1,1},ann.inp.vld{1,2},ann.inp.vld{1,3},ann.inp.vld{1,4},ann.inp.vld{1,5},ann.inp.vld{1,6});
        [ann.out_trn.tst{1,1},ann.out_trn.tst{1,2},ann.out_trn.tst{1,3}] = predict(ann.net,ann.inp.tst{1,1},ann.inp.tst{1,2},ann.inp.tst{1,3},ann.inp.tst{1,4},ann.inp.tst{1,5},ann.inp.tst{1,6});
    elseif index_extra>0 && n_classes>0
        [ann.out_trn.trn{1,1},ann.out_trn.trn{1,2},ann.out_trn.trn{1,3}] = predict(ann.net,ann.inp.trn{1,1},ann.inp.trn{1,2},ann.inp.trn{1,3},ann.inp.trn{1,4},ann.inp.trn{1,5});
        [ann.out_trn.vld{1,1},ann.out_trn.vld{1,2},ann.out_trn.vld{1,3}] = predict(ann.net,ann.inp.vld{1,1},ann.inp.vld{1,2},ann.inp.vld{1,3},ann.inp.vld{1,4},ann.inp.vld{1,5});
        [ann.out_trn.tst{1,1},ann.out_trn.tst{1,2},ann.out_trn.tst{1,3}] = predict(ann.net,ann.inp.tst{1,1},ann.inp.tst{1,2},ann.inp.tst{1,3},ann.inp.tst{1,4},ann.inp.tst{1,5});
    elseif index_extra>0 && n_fm>0
        [ann.out_trn.trn{1,1},ann.out_trn.trn{1,2},ann.out_trn.trn{1,3}] = predict(ann.net,ann.inp.trn{1,1},ann.inp.trn{1,2},ann.inp.trn{1,3},ann.inp.trn{1,4},ann.inp.trn{1,5});
        [ann.out_trn.vld{1,1},ann.out_trn.vld{1,2},ann.out_trn.vld{1,3}] = predict(ann.net,ann.inp.vld{1,1},ann.inp.vld{1,2},ann.inp.vld{1,3},ann.inp.vld{1,4},ann.inp.vld{1,5});
        [ann.out_trn.tst{1,1},ann.out_trn.tst{1,2},ann.out_trn.tst{1,3}] = predict(ann.net,ann.inp.tst{1,1},ann.inp.tst{1,2},ann.inp.tst{1,3},ann.inp.tst{1,4},ann.inp.tst{1,5});
    elseif index_extra>0 && n_classes==0
        [ann.out_trn.trn{1,1},ann.out_trn.trn{1,2},ann.out_trn.trn{1,3}] = predict(ann.net,ann.inp.trn{1,1},ann.inp.trn{1,2},ann.inp.trn{1,3},ann.inp.trn{1,4});
        [ann.out_trn.vld{1,1},ann.out_trn.vld{1,2},ann.out_trn.vld{1,3}] = predict(ann.net,ann.inp.vld{1,1},ann.inp.vld{1,2},ann.inp.vld{1,3},ann.inp.vld{1,4});
        [ann.out_trn.tst{1,1},ann.out_trn.tst{1,2},ann.out_trn.tst{1,3}] = predict(ann.net,ann.inp.tst{1,1},ann.inp.tst{1,2},ann.inp.tst{1,3},ann.inp.tst{1,4});
    else
        [ann.out_trn.trn{1,1},ann.out_trn.trn{1,2},ann.out_trn.trn{1,3}] = predict(ann.net,ann.inp.trn{1,1},ann.inp.trn{1,2},ann.inp.trn{1,3});
        [ann.out_trn.vld{1,1},ann.out_trn.vld{1,2},ann.out_trn.vld{1,3}] = predict(ann.net,ann.inp.vld{1,1},ann.inp.vld{1,2},ann.inp.vld{1,3});
        [ann.out_trn.tst{1,1},ann.out_trn.tst{1,2},ann.out_trn.tst{1,3}] = predict(ann.net,ann.inp.tst{1,1},ann.inp.tst{1,2},ann.inp.tst{1,3});
    end

    if strcmp(TransferLearning,'True')
        if index_extra>0 && n_classes>0 && n_fm>0 && n_rg>0
            [ann.out_trn2.trn{1,1},ann.out_trn2.trn{1,2},ann.out_trn2.trn{1,3}] = predict(ann.net,ann.inp2.trn{1,1},ann.inp2.trn{1,2},ann.inp2.trn{1,3},ann.inp2.trn{1,4},ann.inp2.trn{1,5},ann.inp2.trn{1,6},ann.inp2.trn{1,7});
            [ann.out_trn2.vld{1,1},ann.out_trn2.vld{1,2},ann.out_trn2.vld{1,3}] = predict(ann.net,ann.inp2.vld{1,1},ann.inp2.vld{1,2},ann.inp2.vld{1,3},ann.inp2.vld{1,4},ann.inp2.vld{1,5},ann.inp2.vld{1,6},ann.inp2.vld{1,7});
            [ann.out_trn2.tst{1,1},ann.out_trn2.tst{1,2},ann.out_trn2.tst{1,3}] = predict(ann.net,ann.inp2.tst{1,1},ann.inp2.tst{1,2},ann.inp2.tst{1,3},ann.inp2.tst{1,4},ann.inp2.tst{1,5},ann.inp2.tst{1,6},ann.inp2.tst{1,7});
        elseif index_extra>0 && n_classes>0 && n_fm>0
            [ann.out_trn2.trn{1,1},ann.out_trn2.trn{1,2},ann.out_trn2.trn{1,3}] = predict(ann.net,ann.inp2.trn{1,1},ann.inp2.trn{1,2},ann.inp2.trn{1,3},ann.inp2.trn{1,4},ann.inp2.trn{1,5},ann.inp2.trn{1,6});
            [ann.out_trn2.vld{1,1},ann.out_trn2.vld{1,2},ann.out_trn2.vld{1,3}] = predict(ann.net,ann.inp2.vld{1,1},ann.inp2.vld{1,2},ann.inp2.vld{1,3},ann.inp2.vld{1,4},ann.inp2.vld{1,5},ann.inp2.vld{1,6});
            [ann.out_trn2.tst{1,1},ann.out_trn2.tst{1,2},ann.out_trn2.tst{1,3}] = predict(ann.net,ann.inp2.tst{1,1},ann.inp2.tst{1,2},ann.inp2.tst{1,3},ann.inp2.tst{1,4},ann.inp2.tst{1,5},ann.inp2.tst{1,6});
        elseif index_extra>0 && n_rg>0 && n_fm>0
            [ann.out_trn2.trn{1,1},ann.out_trn2.trn{1,2},ann.out_trn2.trn{1,3}] = predict(ann.net,ann.inp2.trn{1,1},ann.inp2.trn{1,2},ann.inp2.trn{1,3},ann.inp2.trn{1,4},ann.inp2.trn{1,5},ann.inp2.trn{1,6});
            [ann.out_trn2.vld{1,1},ann.out_trn2.vld{1,2},ann.out_trn2.vld{1,3}] = predict(ann.net,ann.inp2.vld{1,1},ann.inp2.vld{1,2},ann.inp2.vld{1,3},ann.inp2.vld{1,4},ann.inp2.vld{1,5},ann.inp2.vld{1,6});
            [ann.out_trn2.tst{1,1},ann.out_trn2.tst{1,2},ann.out_trn2.tst{1,3}] = predict(ann.net,ann.inp2.tst{1,1},ann.inp2.tst{1,2},ann.inp2.tst{1,3},ann.inp2.tst{1,4},ann.inp2.tst{1,5},ann.inp2.tst{1,6});
        elseif index_extra>0 && n_classes>0
            [ann.out_trn2.trn{1,1},ann.out_trn2.trn{1,2},ann.out_trn2.trn{1,3}] = predict(ann.net,ann.inp2.trn{1,1},ann.inp2.trn{1,2},ann.inp2.trn{1,3},ann.inp2.trn{1,4},ann.inp2.trn{1,5});
            [ann.out_trn2.vld{1,1},ann.out_trn2.vld{1,2},ann.out_trn2.vld{1,3}] = predict(ann.net,ann.inp2.vld{1,1},ann.inp2.vld{1,2},ann.inp2.vld{1,3},ann.inp2.vld{1,4},ann.inp2.vld{1,5});
            [ann.out_trn2.tst{1,1},ann.out_trn2.tst{1,2},ann.out_trn2.tst{1,3}] = predict(ann.net,ann.inp2.tst{1,1},ann.inp2.tst{1,2},ann.inp2.tst{1,3},ann.inp2.tst{1,4},ann.inp2.tst{1,5});
        elseif index_extra>0 && n_fm>0
            [ann.out_trn2.trn{1,1},ann.out_trn2.trn{1,2},ann.out_trn2.trn{1,3}] = predict(ann.net,ann.inp2.trn{1,1},ann.inp2.trn{1,2},ann.inp2.trn{1,3},ann.inp2.trn{1,4},ann.inp2.trn{1,5});
            [ann.out_trn2.vld{1,1},ann.out_trn2.vld{1,2},ann.out_trn2.vld{1,3}] = predict(ann.net,ann.inp2.vld{1,1},ann.inp2.vld{1,2},ann.inp2.vld{1,3},ann.inp2.vld{1,4},ann.inp2.vld{1,5});
            [ann.out_trn2.tst{1,1},ann.out_trn2.tst{1,2},ann.out_trn2.tst{1,3}] = predict(ann.net,ann.inp2.tst{1,1},ann.inp2.tst{1,2},ann.inp2.tst{1,3},ann.inp2.tst{1,4},ann.inp2.tst{1,5});
        elseif index_extra>0 && n_classes==0
            [ann.out_trn2.trn{1,1},ann.out_trn2.trn{1,2},ann.out_trn2.trn{1,3}] = predict(ann.net,ann.inp2.trn{1,1},ann.inp2.trn{1,2},ann.inp2.trn{1,3},ann.inp2.trn{1,4});
            [ann.out_trn2.vld{1,1},ann.out_trn2.vld{1,2},ann.out_trn2.vld{1,3}] = predict(ann.net,ann.inp2.vld{1,1},ann.inp2.vld{1,2},ann.inp2.vld{1,3},ann.inp2.vld{1,4});
            [ann.out_trn2.tst{1,1},ann.out_trn2.tst{1,2},ann.out_trn2.tst{1,3}] = predict(ann.net,ann.inp2.tst{1,1},ann.inp2.tst{1,2},ann.inp2.tst{1,3},ann.inp2.tst{1,4});
        else
            [ann.out_trn2.trn{1,1},ann.out_trn2.trn{1,2},ann.out_trn2.trn{1,3}] = predict(ann.net,ann.inp2.trn{1,1},ann.inp2.trn{1,2},ann.inp2.trn{1,3});
            [ann.out_trn2.vld{1,1},ann.out_trn2.vld{1,2},ann.out_trn2.vld{1,3}] = predict(ann.net,ann.inp2.vld{1,1},ann.inp2.vld{1,2},ann.inp2.vld{1,3});
            [ann.out_trn2.tst{1,1},ann.out_trn2.tst{1,2},ann.out_trn2.tst{1,3}] = predict(ann.net,ann.inp2.tst{1,1},ann.inp2.tst{1,2},ann.inp2.tst{1,3});
        end
    end
else %not h12v
     if index_extra>0 && n_classes>0 && n_fm>0 && n_rg>0
        [ann.out_trn.trn{1,1}] = predict(ann.net,ann.inp.trn{1,1},ann.inp.trn{1,2},ann.inp.trn{1,3},ann.inp.trn{1,4},ann.inp.trn{1,5});
        [ann.out_trn.vld{1,1}] = predict(ann.net,ann.inp.vld{1,1},ann.inp.vld{1,2},ann.inp.vld{1,3},ann.inp.vld{1,4},ann.inp.vld{1,5});
        [ann.out_trn.tst{1,1}] = predict(ann.net,ann.inp.tst{1,1},ann.inp.tst{1,2},ann.inp.tst{1,3},ann.inp.tst{1,4},ann.inp.tst{1,5});
    elseif index_extra>0 && n_rg>0 && n_fm>0
        [ann.out_trn.trn{1,1}] = predict(ann.net,ann.inp.trn{1,1},ann.inp.trn{1,2},ann.inp.trn{1,3},ann.inp.trn{1,4});
        [ann.out_trn.vld{1,1}] = predict(ann.net,ann.inp.vld{1,1},ann.inp.vld{1,2},ann.inp.vld{1,3},ann.inp.vld{1,4});
        [ann.out_trn.tst{1,1}] = predict(ann.net,ann.inp.tst{1,1},ann.inp.tst{1,2},ann.inp.tst{1,3},ann.inp.tst{1,4});   
    elseif index_extra>0 && n_classes>0
        [ann.out_trn.trn{1,1}] = predict(ann.net,ann.inp.trn{1,1},ann.inp.trn{1,2},ann.inp.trn{1,3});
        [ann.out_trn.vld{1,1}] = predict(ann.net,ann.inp.vld{1,1},ann.inp.vld{1,2},ann.inp.vld{1,3});
        [ann.out_trn.tst{1,1}] = predict(ann.net,ann.inp.tst{1,1},ann.inp.tst{1,2},ann.inp.tst{1,3});
    elseif index_extra>0 && n_classes==0
        [ann.out_trn.trn{1,1}] = predict(ann.net,ann.inp.trn{1,1},ann.inp.trn{1,2});
        [ann.out_trn.vld{1,1}] = predict(ann.net,ann.inp.vld{1,1},ann.inp.vld{1,2});
        [ann.out_trn.tst{1,1}] = predict(ann.net,ann.inp.tst{1,1},ann.inp.tst{1,2});
    else
        [ann.out_trn.trn{1,1}] = predict(ann.net,ann.inp.trn{1,1});
        [ann.out_trn.vld{1,1}] = predict(ann.net,ann.inp.vld{1,1});
        [ann.out_trn.tst{1,1}] = predict(ann.net,ann.inp.tst{1,1});
    end

    if strcmp(TransferLearning,'True')

        if index_extra>0 && n_classes>0 && n_fm>0 && n_rg>0
            [ann.out_trn2.trn{1,1}] = predict(ann.net,ann.inp2.trn{1,1},ann.inp2.trn{1,2},ann.inp2.trn{1,3},ann.inp2.trn{1,4},ann.inp2.trn{1,5});
            [ann.out_trn2.vld{1,1}] = predict(ann.net,ann.inp2.vld{1,1},ann.inp2.vld{1,2},ann.inp2.vld{1,3},ann.inp2.vld{1,4},ann.inp2.vld{1,5});
            [ann.out_trn2.tst{1,1}] = predict(ann.net,ann.inp2.tst{1,1},ann.inp2.tst{1,2},ann.inp2.tst{1,3},ann.inp2.tst{1,4},ann.inp2.tst{1,5});
        elseif index_extra>0 && n_rg>0 && n_fm>0
            [ann.out_trn2.trn{1,1}] = predict(ann.net,ann.inp2.trn{1,1},ann.inp2.trn{1,2},ann.inp2.trn{1,3},ann.inp2.trn{1,4});
            [ann.out_trn2.vld{1,1}] = predict(ann.net,ann.inp2.vld{1,1},ann.inp2.vld{1,2},ann.inp2.vld{1,3},ann.inp2.vld{1,4});
            [ann.out_trn2.tst{1,1}] = predict(ann.net,ann.inp2.tst{1,1},ann.inp2.tst{1,2},ann.inp2.tst{1,3},ann.inp2.tst{1,4});   
        elseif index_extra>0 && n_classes>0
            [ann.out_trn2.trn{1,1}] = predict(ann.net,ann.inp2.trn{1,1},ann.inp2.trn{1,2},ann.inp2.trn{1,3});
            [ann.out_trn2.vld{1,1}] = predict(ann.net,ann.inp2.vld{1,1},ann.inp2.vld{1,2},ann.inp2.vld{1,3});
            [ann.out_trn2.tst{1,1}] = predict(ann.net,ann.inp2.tst{1,1},ann.inp2.tst{1,2},ann.inp2.tst{1,3});
        elseif index_extra>0 && n_classes==0
            [ann.out_trn2.trn{1,1}] = predict(ann.net,ann.inp2.trn{1,1},ann.inp2.trn{1,2});
            [ann.out_trn2.vld{1,1}] = predict(ann.net,ann.inp2.vld{1,1},ann.inp2.vld{1,2});
            [ann.out_trn2.tst{1,1}] = predict(ann.net,ann.inp2.tst{1,1},ann.inp2.tst{1,2});
        else
            [ann.out_trn2.trn{1,1}] = predict(ann.net,ann.inp2.trn{1,1});
            [ann.out_trn2.vld{1,1}] = predict(ann.net,ann.inp2.vld{1,1});
            [ann.out_trn2.tst{1,1},] = predict(ann.net,ann.inp2.tst{1,1});
        end
    end

end

%% *COMPUTE PERFORMANCE*
fprintf('COMPUTING PERFORMANCE...\n')

if strcmp(component,'h12v')
    if strcmp(TransferLearning,'True')
        ann.prf.trn_rmse = rmse([ann.out_trn2.trn{1,1},ann.out_trn2.trn{1,2},ann.out_trn2.trn{1,3}],[ann.tar2.trn{1,1},ann.tar2.trn{1,2},ann.tar2.trn{1,3}],"all");
        ann.prf.vld_rmse = rmse([ann.out_trn2.vld{1,1},ann.out_trn2.vld{1,2},ann.out_trn2.vld{1,3}],[ann.tar2.vld{1,1},ann.tar2.vld{1,2},ann.tar2.vld{1,3}],"all");
        ann.prf.tst_rmse = rmse([ann.out_trn2.tst{1,1},ann.out_trn2.tst{1,2},ann.out_trn2.tst{1,3}],[ann.tar2.tst{1,1},ann.tar2.tst{1,2},ann.tar2.tst{1,3}],"all");
        % ann.prf.r = (regression(reshape([ann.out_tar2.trn{1,1},ann.out_tar2.trn{1,2},ann.out_tar2.trn{1,3}],1,[]),reshape([ann.out_trn2.trn{1,1},ann.out_trn2.trn{1,2},ann.out_trn2.trn{1,3}],1,[])))^2;
        ann.prf.trn_mae = sum(abs(reshape([ann.tar2.trn{1,1},ann.tar2.trn{1,2},ann.tar2.trn{1,3}],1,[])-reshape([ann.out_trn2.trn{1,1},...
            ann.out_trn2.trn{1,2},ann.out_trn2.trn{1,3}],1,[])))/length(reshape([ann.out_trn2.trn{1,1},ann.out_trn2.trn{1,2},ann.out_trn2.trn{1,3}],1,[]));
         ann.prf.vld_mae = sum(abs(reshape([ann.tar2.vld{1,1},ann.tar2.vld{1,2},ann.tar2.vld{1,3}],1,[])-reshape([ann.out_trn2.vld{1,1},...
            ann.out_trn2.vld{1,2},ann.out_trn2.vld{1,3}],1,[])))/length(reshape([ann.out_trn2.vld{1,1},ann.out_trn2.vld{1,2},ann.out_trn2.vld{1,3}],1,[]));
        ann.prf.tst_mae = sum(abs(reshape([ann.tar2.tst{1,1},ann.tar2.tst{1,2},ann.tar2.tst{1,3}],1,[])-reshape([ann.out_trn2.tst{1,1},...
            ann.out_trn2.tst{1,2},ann.out_trn2.tst{1,3}],1,[])))/length(reshape([ann.out_trn2.tst{1,1},ann.out_trn2.tst{1,2},ann.out_trn2.tst{1,3}],1,[]));
        
        T = reshape([ann.tar2.trn{1,1}, ann.tar2.trn{1,2}, ann.tar2.trn{1,3}], 1, []);
        Y = reshape([ann.out_trn2.trn{1,1}, ann.out_trn2.trn{1,2}, ann.out_trn2.trn{1,3}], 1, []);
        SSres = sum((T - Y).^2);
        SStot = sum((T - mean(T)).^2);
        ann.prf.trn_R2 = 1 - SSres/SStot;
        ann.prf.trn_r = corr(T(:), Y(:), 'rows','complete');

        T = reshape([ann.tar2.vld{1,1}, ann.tar2.vld{1,2}, ann.tar2.vld{1,3}], 1, []);
        Y = reshape([ann.out_trn2.vld{1,1}, ann.out_trn2.vld{1,2}, ann.out_trn2.vld{1,3}], 1, []);
        SSres = sum((T - Y).^2);
        SStot = sum((T - mean(T)).^2);
        ann.prf.vld_R2 = 1 - SSres/SStot;
        ann.prf.vld_r = corr(T(:), Y(:), 'rows','complete');

        T = reshape([ann.tar2.tst{1,1}, ann.tar2.tst{1,2}, ann.tar2.tst{1,3}], 1, []);
        Y = reshape([ann.out_trn2.tst{1,1}, ann.out_trn2.tst{1,2}, ann.out_trn2.tst{1,3}], 1, []);
        SSres = sum((T - Y).^2);
        SStot = sum((T - mean(T)).^2);
        ann.prf.tst_R2 = 1 - SSres/SStot;
        ann.prf.tst_r = corr(T(:), Y(:), 'rows','complete');

        ann.prf.trn_bias = 100 * (exp(mean(ann.out_trn2.trn{1,1}-ann.tar2.trn{1,1},"all"))-1);
        ann.prf.vld_bias = 100 * (exp(mean(ann.out_trn2.vld{1,1}-ann.tar2.vld{1,1},"all"))-1);
        ann.prf.tst_bias = 100 * (exp(mean(ann.out_trn2.tst{1,1}-ann.tar2.tst{1,1},"all"))-1);
    else
        ann.prf.trn_rmse = rmse([ann.out_trn.trn{1,1},ann.out_trn.trn{1,2},ann.out_trn.trn{1,3}],[ann.tar.trn{1,1},ann.tar.trn{1,2},ann.tar.trn{1,3}],"all");
        ann.prf.vld_rmse = rmse([ann.out_trn.vld{1,1},ann.out_trn.vld{1,2},ann.out_trn.vld{1,3}],[ann.tar.vld{1,1},ann.tar.vld{1,2},ann.tar.vld{1,3}],"all");
        ann.prf.tst_rmse = rmse([ann.out_trn.tst{1,1},ann.out_trn.tst{1,2},ann.out_trn.tst{1,3}],[ann.tar.tst{1,1},ann.tar.tst{1,2},ann.tar.tst{1,3}],"all");
        % ann.prf.r = (regression(reshape([ann.out_tar.trn{1,1},ann.out_tar.trn{1,2},ann.out_tar.trn{1,3}],1,[]),reshape([ann.out_trn.trn{1,1},ann.out_trn.trn{1,2},ann.out_trn.trn{1,3}],1,[])))^2;
       ann.prf.trn_mae = sum(abs(reshape([ann.tar.trn{1,1},ann.tar.trn{1,2},ann.tar.trn{1,3}],1,[])-reshape([ann.out_trn.trn{1,1},...
            ann.out_trn.trn{1,2},ann.out_trn.trn{1,3}],1,[])))/length(reshape([ann.out_trn.trn{1,1},ann.out_trn.trn{1,2},ann.out_trn.trn{1,3}],1,[]));
        ann.prf.vld_mae = sum(abs(reshape([ann.tar.vld{1,1},ann.tar.vld{1,2},ann.tar.vld{1,3}],1,[])-reshape([ann.out_trn.vld{1,1},...
            ann.out_trn.vld{1,2},ann.out_trn.vld{1,3}],1,[])))/length(reshape([ann.out_trn.vld{1,1},ann.out_trn.vld{1,2},ann.out_trn.vld{1,3}],1,[]));
        ann.prf.tst_mae = sum(abs(reshape([ann.tar.tst{1,1},ann.tar.tst{1,2},ann.tar.tst{1,3}],1,[])-reshape([ann.out_trn.tst{1,1},...
            ann.out_trn.tst{1,2},ann.out_trn.tst{1,3}],1,[])))/length(reshape([ann.out_trn.tst{1,1},ann.out_trn.tst{1,2},ann.out_trn.tst{1,3}],1,[]));
       
        T = reshape([ann.tar.trn{1,1}, ann.tar.trn{1,2}, ann.tar.trn{1,3}], 1, []);
        Y = reshape([ann.out_trn.trn{1,1}, ann.out_trn.trn{1,2}, ann.out_trn.trn{1,3}], 1, []);
        SSres = sum((T - Y).^2);
        SStot = sum((T - mean(T)).^2);
        ann.prf.trn_R2 = 1 - SSres/SStot;
        ann.prf.trn_r = corr(T(:), Y(:), 'rows','complete');

        T = reshape([ann.tar.vld{1,1}, ann.tar.vld{1,2}, ann.tar.vld{1,3}], 1, []);
        Y = reshape([ann.out_trn.vld{1,1}, ann.out_trn.vld{1,2}, ann.out_trn.vld{1,3}], 1, []);
        SSres = sum((T - Y).^2);
        SStot = sum((T - mean(T)).^2);
        ann.prf.vld_R2 = 1 - SSres/SStot;
        ann.prf.vld_r = corr(T(:), Y(:), 'rows','complete');
        

        T = reshape([ann.tar.tst{1,1}, ann.tar.tst{1,2}, ann.tar.tst{1,3}], 1, []);
        Y = reshape([ann.out_trn.tst{1,1}, ann.out_trn.tst{1,2}, ann.out_trn.tst{1,3}], 1, []);
        SSres = sum((T - Y).^2);
        SStot = sum((T - mean(T)).^2);
        ann.prf.tst_R2 = 1 - SSres/SStot;
        ann.prf.tst_r = corr(T(:), Y(:), 'rows','complete');

        ann.prf.trn_bias = 100 * (exp(mean(ann.out_trn.trn{1,1}-ann.tar.trn{1,1},"all"))-1);
        ann.prf.vld_bias = 100 * (exp(mean(ann.out_trn.vld{1,1}-ann.tar.vld{1,1},"all"))-1);
        ann.prf.tst_bias = 100 * (exp(mean(ann.out_trn.tst{1,1}-ann.tar.tst{1,1},"all"))-1);

        %checking loss
        % Y1 = ann.out_trn.vld{1,1};
        % T1 = ann.tar.vld{1,1};
        %  Y2 = ann.out_trn.vld{1,2};
        % T2 = ann.tar.vld{1,2};
        %  Y3 = ann.out_trn.vld{1,3};
        % T3 = ann.tar.vld{1,3};
        % delta = 0.5;
        % w = ones(size(Y1,2),1);
        % wBias = ones(size(w));
        % wBias(2:7) = 3;
        % wBias(1) = 2;
        % wBias = wBias/mean(wBias);
        % 
        % whuber = @(Y,T,w,delta) sum( w' .* mean( ...
        %     0.5*(abs(Y-T)<=delta).*(Y-T).^2 + ...
        %     (abs(Y-T)>delta).*delta.*(abs(Y-T)-0.5*delta), ...
        %     1) ) ./ sum(w);
        % 
        % wbias = @(Y,T,wBias) sum( wBias' .* (mean(Y-T,1)).^2 ) ./ sum(wBias);
        % 
        % alpha = 10;
        % Lh = 0.375*whuber(Y1,T1,w,delta)+0.375*whuber(Y2,T2,w,delta)+0.25*whuber(Y3,T3,w,delta);
        % Lb = 0.375*wbias(Y1,T1,wBias)+0.375*wbias(Y2,T2,wBias)+0.25*wbias(Y3,T3,wBias);
        % 
        % fprintf('Huber loss      = %.6f\n', Lh);
        % fprintf('Bias loss       = %.6f\n', Lb);
        % fprintf('alpha*bias loss = %.6f\n', alpha*Lb);
        % fprintf('Total loss = %.6f\n', alpha*Lb+Lh);
       
    end
else %one component
    if strcmp(TransferLearning,'True')
        ann.prf.trn_rmse = rmse([ann.tar2.trn{1,1}],[ann.out_trn2.trn{1,1}],"all");
        ann.prf.vld_rmse = rmse([ann.tar2.vld{1,1}],[ann.out_trn2.vld{1,1}],"all");
        ann.prf.tst_rmse = rmse([ann.tar2.tst{1,1}],[ann.out_trn2.tst{1,1}],"all");
        ann.prf.trn_mae = sum(abs(reshape([ann.tar2.trn{1,1}],1,[])-reshape([ann.out_trn2.trn{1,1}],1,[])))/length(reshape([ann.out_trn2.trn{1,1}],1,[]));
        ann.prf.vld_mae = sum(abs(reshape([ann.tar2.vld{1,1}],1,[])-reshape([ann.out_trn2.vld{1,1}],1,[])))/length(reshape([ann.out_trn2.vld{1,1}],1,[]));
        ann.prf.tst_mae = sum(abs(reshape([ann.tar2.tst{1,1}],1,[])-reshape([ann.out_trn2.tst{1,1}],1,[])))/length(reshape([ann.out_trn2.tst{1,1}],1,[]));
        
        T = reshape([ann.tar2.trn{1,1}], 1, []);
        Y = reshape([ann.out_trn2.trn{1,1}], 1, []);
        SSres = sum((T - Y).^2);
        SStot = sum((T - mean(T)).^2);
        ann.prf.trn_R2 = 1 - SSres/SStot;
        ann.prf.trn_r = corr(T(:), Y(:), 'rows','complete');

        T = reshape([ann.tar2.vld{1,1}], 1, []);
        Y = reshape([ann.out_trn2.vld{1,1}], 1, []);
        SSres = sum((T - Y).^2);
        SStot = sum((T - mean(T)).^2);
        ann.prf.vld_R2 = 1 - SSres/SStot;
        ann.prf.vld_r = corr(T(:), Y(:), 'rows','complete');

        T = reshape([ann.tar2.tst{1,1}], 1, []);
        Y = reshape([ann.out_trn2.tst{1,1}], 1, []);
        SSres = sum((T - Y).^2);
        SStot = sum((T - mean(T)).^2);
        ann.prf.tst_R2 = 1 - SSres/SStot;
        ann.prf.tst_r = corr(T(:), Y(:), 'rows','complete');

        ann.prf.trn_bias = 100 * (exp(mean(ann.out_trn2.trn{1,1}-ann.tar2.trn{1,1},"all"))-1);
        ann.prf.vld_bias = 100 * (exp(mean(ann.out_trn2.vld{1,1}-ann.tar2.vld{1,1},"all"))-1);
        ann.prf.tst_bias = 100 * (exp(mean(ann.out_trn2.tst{1,1}-ann.tar2.tst{1,1},"all"))-1);
    else
        ann.prf.trn_rmse = rmse([ann.tar.trn{1,1}],[ann.out_trn.trn{1,1}],"all");
        ann.prf.vld_rmse = rmse([ann.tar.vld{1,1}],[ann.out_trn.vld{1,1}],"all");
        ann.prf.tst_rmse = rmse([ann.tar.tst{1,1}],[ann.out_trn.tst{1,1}],"all");
        ann.prf.trn_mae = sum(abs(reshape([ann.tar.trn{1,1}],1,[])-reshape([ann.out_trn.trn{1,1}],1,[])))/length(reshape([ann.out_trn.trn{1,1}],1,[]));
        ann.prf.vld_mae = sum(abs(reshape([ann.tar.vld{1,1}],1,[])-reshape([ann.out_trn.vld{1,1}],1,[])))/length(reshape([ann.out_trn.vld{1,1}],1,[]));
        ann.prf.tst_mae = sum(abs(reshape([ann.tar.tst{1,1}],1,[])-reshape([ann.out_trn.tst{1,1}],1,[])))/length(reshape([ann.out_trn.tst{1,1}],1,[]));
    
        T = reshape([ann.tar.trn{1,1}], 1, []);
        Y = reshape([ann.out_trn.trn{1,1}], 1, []);
        SSres = sum((T - Y).^2);
        SStot = sum((T - mean(T)).^2);
        ann.prf.trn_R2 = 1 - SSres/SStot;
        ann.prf.trn_r = corr(T(:), Y(:), 'rows','complete');

        T = reshape([ann.tar.vld{1,1}], 1, []);
        Y = reshape([ann.out_trn.vld{1,1}], 1, []);
        SSres = sum((T - Y).^2);
        SStot = sum((T - mean(T)).^2);
        ann.prf.vld_R2 = 1 - SSres/SStot;
        ann.prf.vld_r = corr(T(:), Y(:), 'rows','complete');

        T = reshape([ann.tar.tst{1,1}], 1, []);
        Y = reshape([ann.out_trn.tst{1,1}], 1, []);
        SSres = sum((T - Y).^2);
        SStot = sum((T - mean(T)).^2);
        ann.prf.tst_R2 = 1 - SSres/SStot;
        ann.prf.tst_r = corr(T(:), Y(:), 'rows','complete');

        ann.prf.trn_bias = 100 * (exp(mean(ann.out_trn.trn{1,1}-ann.tar.trn{1,1},"all"))-1);
        ann.prf.vld_bias = 100 * (exp(mean(ann.out_trn.vld{1,1}-ann.tar.vld{1,1},"all"))-1);
        ann.prf.tst_bias = 100 * (exp(mean(ann.out_trn.tst{1,1}-ann.tar.tst{1,1},"all"))-1);
    end

end

%% *OUTPUT*
varargout{1} = ann;
return
end
