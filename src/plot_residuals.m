function plot_residuals(varargin)
%% *SET-UP*
ann = varargin{1};
inp = varargin{2};
tar = varargin{3};
wd  = varargin{4};
dsg = varargin{5};
dbn_name = varargin{6};
verNet = varargin{7};
component= varargin{8};
TransferLearning = varargin{9};

add_distance = varargin{10};
add_m = varargin{11};
add_lndistance = varargin{12};
separate_classes = varargin{13};
add_vs30 = varargin{14};
add_fm = varargin{15};
add_rg = varargin{16};
add_depth = varargin{17};
ann2 = varargin{18};

plot_set_up;
TnC  = inp.vTn(1);
xpl = cell(3,1);
ypl = cell(3,1);
nT=length(tar.vTn(:));
%% *DEFINE LIMITS*
% _TRAINING SET_
% xlm =  [0.00;1.00];
ylm =  [-1.1;1.1];
xtk =  0.00:0.25:1.00;
ytk = -1.00:0.25:1.00;
xlm = [-0.05;1];

%% *COMPUTE ERROR BARS*
% _TRAINING SET_
%
xpl{2,1} = tar.vTn(:)./TnC;
xpl{3,1} = tar.vTn(:)./TnC;
xpl{4,1} = tar.vTn(:)./TnC;
%
%Perfomance plots
    if strcmp(TransferLearning,'True')
        ypl{2,1} = mean([ann.out_trn2.trn{1,1}]-[ann.tar2.trn{1,1}],1);
        ypl{3,1} = mean([ann.out_trn2.vld{1,1}]-[ann.tar2.vld{1,1}],1);
        ypl{4,1} = mean([ann.out_trn2.tst{1,1}]-[ann.tar2.tst{1,1}],1);

        err{2,1}(:,1) = prctile([ann.out_trn2.trn{1,1}]-[ann.tar2.trn{1,1}],16,1);
        err{2,1}(:,2) = prctile([ann.out_trn2.trn{1,1}]-[ann.tar2.trn{1,1}],84,1);

        err{3,1}(:,1) = prctile([ann.out_trn2.vld{1,1}]-[ann.tar2.vld{1,1}],16,1);
        err{3,1}(:,2) = prctile([ann.out_trn2.vld{1,1}]-[ann.tar2.vld{1,1}],84,1);

        err{4,1}(:,1) = prctile([ann.out_trn2.tst{1,1}]-[ann.tar2.tst{1,1}],16,1);
        err{4,1}(:,2) = prctile([ann.out_trn2.tst{1,1}]-[ann.tar2.tst{1,1}],84,1);

        % ypl{2,1} = mean([ann.out_trn2.trn{1,1},ann.out_trn2.trn{1,2},ann.out_trn2.trn{1,3}]-[ann.tar2.trn{1,1},ann.tar2.trn{1,2},ann.tar2.trn{1,3}],1);
        % ypl{3,1} = mean([ann.out_trn2.vld{1,1},ann.out_trn2.vld{1,2},ann.out_trn2.vld{1,3}]-[ann.tar2.vld{1,1},ann.tar2.vld{1,2},ann.tar2.vld{1,3}],1);
        % ypl{4,1} = mean([ann.out_trn2.tst{1,1},ann.out_trn2.tst{1,2},ann.out_trn2.tst{1,3}]-[ann.tar2.tst{1,1},ann.tar2.tst{1,2},ann.tar2.tst{1,3}],1);
        % 
        % err{2,1}(:,1) = prctile([ann.out_trn2.trn{1,1},ann.out_trn2.trn{1,2},ann.out_trn2.trn{1,3}]-[ann.tar2.trn{1,1},ann.tar2.trn{1,2},ann.tar2.trn{1,3}],16,1);
        % err{2,1}(:,2) = prctile([ann.out_trn2.trn{1,1},ann.out_trn2.trn{1,2},ann.out_trn2.trn{1,3}]-[ann.tar2.trn{1,1},ann.tar2.trn{1,2},ann.tar2.trn{1,3}],84,1);
        % 
        % err{3,1}(:,1) = prctile([ann.out_trn2.vld{1,1},ann.out_trn2.vld{1,2},ann.out_trn2.vld{1,3}]-[ann.tar2.vld{1,1},ann.tar2.vld{1,2},ann.tar2.vld{1,3}],16,1);
        % err{3,1}(:,2) = prctile([ann.out_trn2.vld{1,1},ann.out_trn2.vld{1,2},ann.out_trn2.vld{1,3}]-[ann.tar2.vld{1,1},ann.tar2.vld{1,2},ann.tar2.vld{1,3}],84,1);
        % 
        % err{4,1}(:,1) = prctile([ann.out_trn2.tst{1,1},ann.out_trn2.tst{1,2},ann.out_trn2.tst{1,3}]-[ann.tar2.tst{1,1},ann.tar2.tst{1,2},ann.tar2.tst{1,3}],16,1);
        % err{4,1}(:,2) = prctile([ann.out_trn2.tst{1,1},ann.out_trn2.tst{1,2},ann.out_trn2.tst{1,3}]-[ann.tar2.tst{1,1},ann.tar2.tst{1,2},ann.tar2.tst{1,3}],84,1);
    else
        ypl{2,1} = mean([ann.out_trn.trn{1,1}]-[ann.tar.trn{1,1}],1);
        ypl{3,1} = mean([ann.out_trn.vld{1,1}]-[ann.tar.vld{1,1}],1);
        ypl{4,1} = mean([ann.out_trn.tst{1,1}]-[ann.tar.tst{1,1}],1);

        err{2,1}(:,1) = prctile([ann.out_trn.trn{1,1}]-[ann.tar.trn{1,1}],16,1);
        err{2,1}(:,2) = prctile([ann.out_trn.trn{1,1}]-[ann.tar.trn{1,1}],84,1);

        err{3,1}(:,1) = prctile([ann.out_trn.vld{1,1}]-[ann.tar.vld{1,1}],16,1);
        err{3,1}(:,2) = prctile([ann.out_trn.vld{1,1}]-[ann.tar.vld{1,1}],84,1);

        err{4,1}(:,1) = prctile([ann.out_trn.tst{1,1}]-[ann.tar.tst{1,1}],16,1);
        err{4,1}(:,2) = prctile([ann.out_trn.tst{1,1}]-[ann.tar.tst{1,1}],84,1);
    end

figure('position',[0,0,14,10]);

pl11=plot(xpl{2,1},ypl{2,1}(1:nT)); hold all;
%     pl11.LineWidth=4;
%     pl11.Color=rgb('lightgrey');
pl11.LineStyle='none';
pl21=bar(xpl{2,1}([1,3,5:nT]),err{2,1}([1,3,5:nT],1)); hold all;
pl3=bar(xpl{2,1}([1,3,5:nT]),err{2,1}([1,3,5:nT],2)); hold all;
pl21.BarWidth=0.9;
pl3.BarWidth=0.9;

pl21.FaceColor=rgb('lightgrey');
pl3.FaceColor=rgb('lightgrey');

pl22=plot(xpl{3,1},ypl{3,1}(1:nT)); hold all;
%     pl22.LineWidth=4;
%     pl22.Color=[0.4,0.4,0.4];
pl22.LineStyle='none';
pl23=bar(xpl{3,1}([1,3,5:nT]),err{3,1}([1,3,5:nT],1)); hold all;
pl3=bar(xpl{3,1}([1,3,5:nT]),err{3,1}([1,3,5:nT],2)); hold all;
pl23.BarWidth=0.5;
pl3.BarWidth=0.5;
pl23.FaceColor=[0.4,0.4,0.4];
pl3.FaceColor=[0.4,0.4,0.4];

pl33=plot(xpl{4,1},ypl{4,1}(1:nT)); hold all;
pl33.LineStyle='none';
%     pl33.LineWidth=4;
%     pl33.Color=rgb('black');
pl24=bar(xpl{4,1}([1,3,5:nT]),err{4,1}([1,3,5:nT],1)); hold all;
pl3=bar(xpl{4,1}([1,3,5:nT]),err{4,1}([1,3,5:nT],2)); hold all;
pl24.BarWidth=0.2;
pl3.BarWidth=0.2;
pl24.FaceColor=rgb('black');
pl3.FaceColor=rgb('black');

xlim(gca,xlm);
ylim(gca,ylm);
set(gca,'xtick',xtk,'ytick',ytk,'linewidth',2);
set(gca,'ticklength',[.01,.01]);
xlabel(gca,'T/T*','fontsize',15,'fontweight','bold');
ylabel(gca,'ln (Sa_{ANN}/Sa_{Obs})','fontsize',15,'fontweight','bold');
leg=legend(gca,[pl21,pl23,pl24],{'TRN';'VLD';'TST'});

set(leg,'interpreter','latex','location','northeast',...
    'orientation','horizontal','box','off','fontsize',15);

set(gca,'fontsize',14);

text(0.7,-0.8,strcat('$T^\star=$',num2str(TnC,'%.2f'),'$s$'),'parent',gca,...
    'interpreter','latex','fontsize',16)
rule_fig(gcf);

if strcmp(TransferLearning,'True')
    if strcmp(add_distance,'True') && strcmp(add_m,'True') && strcmp(add_lndistance,'True') && strcmp(separate_classes,'True') && strcmp(add_fm,'True') && strcmp(add_rg,'True')  && strcmp(add_depth,'True')
    saveas(gcf,fullfile(wd,sprintf('net _%u_%s_%s_%s_%s_%s.jpg',...
        round(ann2.TnC*100) ,ann2.cp,'Rjb_Mw_logRjb_Dh_Site_SoF_Reg',dbn_name,'TL',verNet)));
    elseif strcmp(add_distance,'True') && strcmp(add_m,'True') && strcmp(add_lndistance,'True') && strcmp(add_vs30,'True') && strcmp(add_fm,'True') && strcmp(add_rg,'True') && strcmp(add_depth,'True')
        saveas(gcf,fullfile(wd,sprintf('net _%u_%s_%s_%s_%s_%s.jpg',...
            round(ann2.TnC*100) ,ann2.cp,'Rjb_Mw_logRjb_Vs30_Dh_SoF_Reg',dbn_name,verNet)));
    elseif strcmp(add_distance,'True') && strcmp(add_m,'True') && strcmp(add_lndistance,'True') && strcmp(separate_classes,'True') && strcmp(add_fm,'True') && strcmp(add_rg,'True')
        saveas(gcf,fullfile(wd,sprintf('net _%u_%s_%s_%s_%s_%s.jpg',...
            round(ann2.TnC*100) ,ann2.cp,'Rjb_Mw_logRjb_Site_SoF_Reg',dbn_name,'TL',verNet)));
    elseif strcmp(add_distance,'True') && strcmp(add_m,'True') && strcmp(add_lndistance,'True') && strcmp(separate_classes,'True') && strcmp(add_fm,'True') && strcmp(add_depth,'True')
        saveas(gcf,fullfile(wd,sprintf('net _%u_%s_%s_%s_%s_%s.jpg',...
            round(ann2.TnC*100) ,ann2.cp,'Rjb_Mw_logRjb_Dh_Site_SoF',dbn_name,'TL',verNet)));
    elseif strcmp(add_distance,'True') && strcmp(add_m,'True') && strcmp(add_lndistance,'True') && strcmp(add_vs30,'True') && strcmp(add_fm,'True') && strcmp(add_rg,'True')
        saveas(gcf,fullfile(wd,sprintf('net _%u_%s_%s_%s_%s_%s.jpg',...
            round(ann2.TnC*100) ,ann2.cp,'Rjb_Mw_logRjb_Vs30_SoF_Reg',dbn_name,'TL',verNet)));
    elseif strcmp(add_distance,'True') && strcmp(add_m,'True') && strcmp(add_lndistance,'True') && strcmp(separate_classes,'True') && strcmp(add_fm,'True')
        saveas(gcf,fullfile(wd,sprintf('net _%u_%s_%s_%s_%s_%s.jpg',...
            round(ann2.TnC*100) ,ann2.cp,'Rjb_Mw_logRjb_Site_SoF',dbn_name,'TL',verNet)));
    elseif strcmp(add_distance,'True') && strcmp(add_m,'True') && strcmp(add_lndistance,'True') && strcmp(add_rg,'True') && strcmp(add_fm,'True')
        saveas(gcf,fullfile(wd,sprintf('net _%u_%s_%s_%s_%s_%s.jpg',...
            round(ann2.TnC*100) ,ann2.cp,'Rjb_Mw_logRjb_SoF_Reg',dbn_name,'TL',verNet)));
    elseif strcmp(add_distance,'True') && strcmp(add_m,'True') && strcmp(add_lndistance,'True') && strcmp(add_vs30,'True') && strcmp(add_fm,'True')
        saveas(gcf,fullfile(wd,sprintf('net _%u_%s_%s_%s_%s_%s.jpg',...
            round(ann2.TnC*100) ,ann2.cp,'Rjb_Mw_logRjb_Vs30_SoF',dbn_name,'TL',verNet)));
    elseif strcmp(add_distance,'True') && strcmp(add_m,'True') && strcmp(add_lndistance,'True') && strcmp(separate_classes,'True')
        saveas(gcf,fullfile(wd,sprintf('net _%u_%s_%s_%s_%s_%s.jpg',...
            round(ann2.TnC*100) ,ann2.cp,'Rjb_Mw_logRjb_Site',dbn_name,'TL',verNet)));
    elseif strcmp(add_distance,'True') && strcmp(add_m,'True') && strcmp(add_lndistance,'True') && strcmp(add_fm,'True')
        saveas(gcf,fullfile(wd,sprintf('net _%u_%s_%s_%s_%s_%s.jpg',...
            round(ann2.TnC*100) ,ann2.cp,'Rjb_Mw_logRjb_SoF',dbn_name,'TL',verNet)));
    elseif strcmp(add_distance,'True') && strcmp(add_m,'True') && strcmp(add_lndistance,'True')
        saveas(gcf,fullfile(wd,sprintf('net _%u_%s_%s_%s_%s_%s.jpg',...
            round(ann2.TnC*100) ,ann2.cp,'Rjb_Mw_logRjb',dbn_name,'TL',verNet)));
    elseif strcmp(add_distance,'True') && strcmp(add_m,'True')
        saveas(gcf,fullfile(wd,sprintf('net _%u_%s_%s_%s_%s_%s.jpg',...
            round(ann2.TnC*100) ,ann2.cp,'Rjb_Mw',dbn_name,'TL',verNet)));
    elseif strcmp(add_distance,'True') && strcmp(add_lndistance,'True')
        saveas(gcf,fullfile(wd,sprintf('net _%u_%s_%s_%s_%s_%s.jpg',...
            round(ann2.TnC*100) ,ann2.cp,'Rjb_logRjb',dbn_name,verNet)));
    elseif strcmp(add_m,'True') && strcmp(add_lndistance,'True')
        saveas(gcf,fullfile(wd,sprintf('net _%u_%s_%s_%s_%s_%s.jpg',...
            round(ann2.TnC*100) ,ann2.cp,'Mw_logRjb',dbn_name,verNet)));
    elseif strcmp(add_distance,'True')
        saveas(gcf,fullfile(wd,sprintf('net _%u_%s_%s_%s_%s_%s.jpg',...
            round(ann2.TnC*100) ,ann2.cp,'Rjb',dbn_name,verNet)));
    elseif strcmp(add_lndistance,'True')
        saveas(gcf,fullfile(wd,sprintf('net _%u_%s_%s_%s_%s_%s.jpg',...
            round(ann2.TnC*100) ,ann2.cp,'logRjb',dbn_name,verNet)));
    elseif strcmp(add_m,'True')
        saveas(gcf,fullfile(wd,sprintf('net _%u_%s_%s_%s_%s_%s.jpg',...
            round(ann2.TnC*100) ,ann2.cp,'Mw',dbn_name,verNet)));
    else
        saveas(gcf,fullfile(wd,sprintf('net _%u_%s_%s_%s_%s.jpg',...
            round(ann2.TnC*100) ,ann2.cp,dbn_name,verNet)));
    end
else
    if strcmp(add_distance,'True') && strcmp(add_m,'True') && strcmp(add_lndistance,'True') && strcmp(separate_classes,'True') && strcmp(add_fm,'True') && strcmp(add_rg,'True')  && strcmp(add_depth,'True')
        saveas(gcf,fullfile(wd,sprintf('net _%u_%s_%s_%s_%s.jpg',...
            round(ann2.TnC*100) ,ann2.cp,'Rjb_Mw_logRjb_Dh_Site_SoF_Reg',dbn_name,verNet)));
    elseif strcmp(add_distance,'True') && strcmp(add_m,'True') && strcmp(add_lndistance,'True') && strcmp(add_vs30,'True') && strcmp(add_fm,'True') && strcmp(add_rg,'True') && strcmp(add_depth,'True')
        saveas(gcf,fullfile(wd,sprintf('net _%u_%s_%s_%s_%s.jpg',...
            round(ann2.TnC*100) ,ann2.cp,'Rjb_Mw_logRjb_Vs30_Dh_SoF_Reg',dbn_name,verNet)));
    elseif strcmp(add_distance,'True') && strcmp(add_m,'True') && strcmp(add_lndistance,'True') && strcmp(separate_classes,'True') && strcmp(add_fm,'True') && strcmp(add_rg,'True')
        saveas(gcf,fullfile(wd,sprintf('net _%u_%s_%s_%s_%s.jpg',...
            round(ann2.TnC*100) ,ann2.cp,'Rjb_Mw_logRjb_Site_SoF_Reg',dbn_name,verNet)));
    elseif strcmp(add_distance,'True') && strcmp(add_m,'True') && strcmp(add_lndistance,'True') && strcmp(separate_classes,'True') && strcmp(add_fm,'True') && strcmp(add_depth,'True')
        saveas(gcf,fullfile(wd,sprintf('net _%u_%s_%s_%s_%s.jpg',...
            round(ann2.TnC*100) ,ann2.cp,'Rjb_Mw_logRjb_Dh_Site_SoF',dbn_name,verNet)));
    elseif strcmp(add_distance,'True') && strcmp(add_m,'True') && strcmp(add_lndistance,'True') && strcmp(add_vs30,'True') && strcmp(add_fm,'True') && strcmp(add_rg,'True')
        saveas(gcf,fullfile(wd,sprintf('net _%u_%s_%s_%s_%s.jpg',...
            round(ann2.TnC*100) ,ann2.cp,'Rjb_Mw_logRjb_Vs30_SoF_Reg',dbn_name,verNet)));
    elseif strcmp(add_distance,'True') && strcmp(add_m,'True') && strcmp(add_lndistance,'True') && strcmp(separate_classes,'True') && strcmp(add_fm,'True')
        saveas(gcf,fullfile(wd,sprintf('net _%u_%s_%s_%s_%s.jpg',...
            round(ann2.TnC*100) ,ann2.cp,'Rjb_Mw_logRjb_Site_SoF',dbn_name,verNet)));
    elseif strcmp(add_distance,'True') && strcmp(add_m,'True') && strcmp(add_lndistance,'True') && strcmp(add_rg,'True') && strcmp(add_fm,'True')
        saveas(gcf,fullfile(wd,sprintf('net _%u_%s_%s_%s_%s.jpg',...
            round(ann2.TnC*100) ,ann2.cp,'Rjb_Mw_logRjb_SoF_Reg',dbn_name,verNet)));
    elseif strcmp(add_distance,'True') && strcmp(add_m,'True') && strcmp(add_lndistance,'True') && strcmp(add_vs30,'True') && strcmp(add_fm,'True')
        saveas(gcf,fullfile(wd,sprintf('net _%u_%s_%s_%s_%s.jpg',...
            round(ann2.TnC*100) ,ann2.cp,'Rjb_Mw_logRjb_Vs30_SoF',dbn_name,verNet)));
    elseif strcmp(add_distance,'True') && strcmp(add_m,'True') && strcmp(add_lndistance,'True') && strcmp(separate_classes,'True')
        saveas(gcf,fullfile(wd,sprintf('net _%u_%s_%s_%s_%s.jpg',...
            round(ann2.TnC*100) ,ann2.cp,'Rjb_Mw_logRjb_Site',dbn_name,verNet)));
    elseif strcmp(add_distance,'True') && strcmp(add_m,'True') && strcmp(add_lndistance,'True') && strcmp(add_fm,'True')
        saveas(gcf,fullfile(wd,sprintf('net _%u_%s_%s_%s_%s.jpg',...
            round(ann2.TnC*100) ,ann2.cp,'Rjb_Mw_logRjb_SoF',dbn_name,verNet)));
    elseif strcmp(add_distance,'True') && strcmp(add_m,'True') && strcmp(add_lndistance,'True')
        saveas(gcf,fullfile(wd,sprintf('net _%u_%s_%s_%s_%s.jpg',...
            round(ann2.TnC*100) ,ann2.cp,'Rjb_Mw_logRjb',dbn_name,verNet)));
    elseif strcmp(add_distance,'True') && strcmp(add_m,'True')
        saveas(gcf,fullfile(wd,sprintf('net _%u_%s_%s_%s_%s.jpg',...
            round(ann2.TnC*100) ,ann2.cp,'Rjb_Mw',dbn_name,verNet)));
    elseif strcmp(add_distance,'True') && strcmp(add_lndistance,'True')
        saveas(gcf,fullfile(wd,sprintf('net _%u_%s_%s_%s_%s.jpg',...
            round(ann2.TnC*100) ,ann2.cp,'Rjb_logRjb',dbn_name,verNet)));
    elseif strcmp(add_m,'True') && strcmp(add_lndistance,'True')
        saveas(gcf,fullfile(wd,sprintf('net _%u_%s_%s_%s_%s.jpg',...
            round(ann2.TnC*100) ,ann2.cp,'Mw_logRjb',dbn_name,verNet)));
    elseif strcmp(add_distance,'True')
        saveas(gcf,fullfile(wd,sprintf('net _%u_%s_%s_%s_%s.jpg',...
            round(ann2.TnC*100) ,ann2.cp,'Rjb',dbn_name,verNet)));
    elseif strcmp(add_lndistance,'True')
        saveas(gcf,fullfile(wd,sprintf('net _%u_%s_%s_%s_%s.jpg',...
            round(ann2.TnC*100) ,ann2.cp,'logRjb',dbn_name,verNet)));
    elseif strcmp(add_m,'True')
        saveas(gcf,fullfile(wd,sprintf('net _%u_%s_%s_%s_%s.jpg',...
            round(ann2.TnC*100) ,ann2.cp,'Mw',dbn_name,verNet)));
    else
        saveas(gcf,fullfile(wd,sprintf('net _%u_%s_%s_%s.jpg',...
            round(ann2.TnC*100) ,ann2.cp,dbn_name,verNet)));
    end
end
return
end
