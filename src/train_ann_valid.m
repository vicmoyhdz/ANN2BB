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
    if index_extra>0 && n_classes>0
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

        if index_extra>0 && n_classes>0
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
        ann.prf.trn = rmse([ann.out_trn2.trn{1,1},ann.out_trn2.trn{1,2},ann.out_trn2.trn{1,3}],[ann.tar2.trn{1,1},ann.tar2.trn{1,2},ann.tar2.trn{1,3}],"all");
        ann.prf.vld = rmse([ann.out_trn2.vld{1,1},ann.out_trn2.vld{1,2},ann.out_trn2.vld{1,3}],[ann.tar2.vld{1,1},ann.tar2.vld{1,2},ann.tar2.vld{1,3}],"all");
        ann.prf.tst = rmse([ann.out_trn2.tst{1,1},ann.out_trn2.tst{1,2},ann.out_trn2.tst{1,3}],[ann.tar2.tst{1,1},ann.tar2.tst{1,2},ann.tar2.tst{1,3}],"all");
        % ann.prf.r = (regression(reshape([ann.out_tar2.trn{1,1},ann.out_tar2.trn{1,2},ann.out_tar2.trn{1,3}],1,[]),reshape([ann.out_trn2.trn{1,1},ann.out_trn2.trn{1,2},ann.out_trn2.trn{1,3}],1,[])))^2;
        ann.prf.mae = sum(abs(reshape([ann.tar2.trn{1,1},ann.tar2.trn{1,2},ann.tar2.trn{1,3}],1,[])-reshape([ann.out_trn2.trn{1,1},...
            ann.out_trn2.trn{1,2},ann.out_trn2.trn{1,3}],1,[])))/length(reshape([ann.out_trn2.trn{1,1},ann.out_trn2.trn{1,2},ann.out_trn2.trn{1,3}],1,[]));
        T = reshape([ann.tar2.vld{1,1}, ann.tar2.vld{1,2}, ann.tar2.vld{1,3}], 1, []);
        Y = reshape([ann.out_trn2.vld{1,1}, ann.out_trn2.vld{1,2}, ann.out_trn2.vld{1,3}], 1, []);
        SSres = sum((T - Y).^2);
        SStot = sum((T - mean(T)).^2);
        ann.prf.R2 = 1 - SSres/SStot;
        ann.prf.r = corr(T(:), Y(:), 'rows','complete');
    else
        ann.prf.trn = rmse([ann.out_trn.trn{1,1},ann.out_trn.trn{1,2},ann.out_trn.trn{1,3}],[ann.tar.trn{1,1},ann.tar.trn{1,2},ann.tar.trn{1,3}],"all");
        ann.prf.vld = rmse([ann.out_trn.vld{1,1},ann.out_trn.vld{1,2},ann.out_trn.vld{1,3}],[ann.tar.vld{1,1},ann.tar.vld{1,2},ann.tar.vld{1,3}],"all");
        ann.prf.tst = rmse([ann.out_trn.tst{1,1},ann.out_trn.tst{1,2},ann.out_trn.tst{1,3}],[ann.tar.tst{1,1},ann.tar.tst{1,2},ann.tar.tst{1,3}],"all");
        % ann.prf.r = (regression(reshape([ann.out_tar.trn{1,1},ann.out_tar.trn{1,2},ann.out_tar.trn{1,3}],1,[]),reshape([ann.out_trn.trn{1,1},ann.out_trn.trn{1,2},ann.out_trn.trn{1,3}],1,[])))^2;
        ann.prf.mae = sum(abs(reshape([ann.tar.vld{1,1},ann.tar.vld{1,2},ann.tar.vld{1,3}],1,[])-reshape([ann.out_trn.vld{1,1},...
            ann.out_trn.vld{1,2},ann.out_trn.vld{1,3}],1,[])))/length(reshape([ann.out_trn.vld{1,1},ann.out_trn.vld{1,2},ann.out_trn.vld{1,3}],1,[]));
        T = reshape([ann.tar.vld{1,1}, ann.tar.vld{1,2}, ann.tar.vld{1,3}], 1, []);
        Y = reshape([ann.out_trn.vld{1,1}, ann.out_trn.vld{1,2}, ann.out_trn.vld{1,3}], 1, []);
        SSres = sum((T - Y).^2);
        SStot = sum((T - mean(T)).^2);
        ann.prf.R2 = 1 - SSres/SStot;
        ann.prf.r = corr(T(:), Y(:), 'rows','complete');
    end
else %not h12v
    if strcmp(TransferLearning,'True')
        ann.prf.trn = mse([ann.tar2.trn{1,1}],[ann.out_trn2.trn{1,1}]);
        ann.prf.vld = mse([ann.tar2.vld{1,1}],[ann.out_trn2.vld{1,1}]);
        ann.prf.tst = mse([ann.tar2.tst{1,1}],[ann.out_trn2.tst{1,1}]);
        ann.prf.r = (regression(reshape([ann.tar2.trn{1,1}],1,[]),reshape([ann.out_trn2.trn{1,1}],1,[])))^2;
        ann.prf.mae = sum(abs(reshape([ann.tar2.trn{1,1}],1,[])-reshape([ann.out_trn2.trn{1,1}],1,[])))/length(reshape([ann.out_trn2.trn{1,1}],1,[]));
    else
        ann.prf.trn = mse([ann.tar.trn{1,1}],[ann.out_trn.trn{1,1}]);
        ann.prf.vld = mse([ann.tar.vld{1,1}],[ann.out_trn.vld{1,1}]);
        ann.prf.tst = mse([ann.tar.tst{1,1}],[ann.out_trn.tst{1,1}]);
        ann.prf.r = (regression(reshape([ann.tar.trn{1,1}],1,[]),reshape([ann.out_trn.trn{1,1}],1,[])))^2;
        ann.prf.mae = sum(abs(reshape([ann.tar.trn{1,1}],1,[])-reshape([ann.out_trn.trn{1,1}],1,[])))/length(reshape([ann.out_trn.trn{1,1}],1,[]));
    end

end

%% *OUTPUT*
varargout{1} = ann;
return
end
