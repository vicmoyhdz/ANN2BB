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
            dsT1Trn_1 = arrayDatastore([tar.DATABASE_1(:,dsg.idx.trn);wRecTrain(dsg.idx.trn)]');
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
            dsT1vld_1 = arrayDatastore([tar.DATABASE_1(:,dsg.idx.vld);wRecTrain(dsg.idx.vld)]');
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
    else % only one component

        if index_extra>0 && n_classes>0 && n_fm>0 && n_rg>0
            NNs{i_}.inp.trn = {inp.DATABASE_1(:,dsg.idx.trn)',inp.DATABASE_3(:,dsg.idx.trn)',inp.DATABASE_4(:,dsg.idx.trn)',inp.DATABASE_6(:,dsg.idx.trn)',inp.DATABASE_7(:,dsg.idx.trn)'};
            NNs{i_}.tar.trn = {tar.DATABASE_1(:,dsg.idx.trn)'};
            NNs{i_}.inp.vld = {inp.DATABASE_1(:,dsg.idx.vld)',inp.DATABASE_3(:,dsg.idx.vld)',inp.DATABASE_4(:,dsg.idx.vld)',inp.DATABASE_6(:,dsg.idx.vld)',inp.DATABASE_7(:,dsg.idx.vld)'};
            NNs{i_}.tar.vld = {tar.DATABASE_1(:,dsg.idx.vld)'};
            NNs{i_}.inp.tst = {inp.DATABASE_1(:,dsg.idx.tst)',inp.DATABASE_3(:,dsg.idx.tst)',inp.DATABASE_4(:,dsg.idx.tst)',inp.DATABASE_6(:,dsg.idx.tst)',inp.DATABASE_7(:,dsg.idx.tst)'};
            NNs{i_}.tar.tst = {tar.DATABASE_1(:,dsg.idx.tst)'};

            dsX1Trn_1 = arrayDatastore(inp.DATABASE_1(:,dsg.idx.trn)');
            dsX1Trn_3 = arrayDatastore(inp.DATABASE_3(:,dsg.idx.trn)');
            dsX1Trn_4 = arrayDatastore(inp.DATABASE_4(:,dsg.idx.trn)');
            dsX1Trn_6 = arrayDatastore(inp.DATABASE_6(:,dsg.idx.trn)');
            dsX1Trn_7 = arrayDatastore(inp.DATABASE_7(:,dsg.idx.trn)');
            dsT1Trn_1 = arrayDatastore([tar.DATABASE_1(:,dsg.idx.trn);wRecTrain(dsg.idx.trn)]');
            dsTrn1 = combine(dsX1Trn_1,dsX1Trn_3,dsX1Trn_4,dsX1Trn_6,dsX1Trn_7,dsT1Trn_1);

            dsX1vld_1 = arrayDatastore(inp.DATABASE_1(:,dsg.idx.vld)');
            dsX1vld_3 = arrayDatastore(inp.DATABASE_3(:,dsg.idx.vld)');
            dsX1vld_4 = arrayDatastore(inp.DATABASE_4(:,dsg.idx.vld)');
            dsX1vld_6 = arrayDatastore(inp.DATABASE_6(:,dsg.idx.vld)');
            dsX1vld_7 = arrayDatastore(inp.DATABASE_7(:,dsg.idx.vld)');
            dsT1vld_1 = arrayDatastore([tar.DATABASE_1(:,dsg.idx.vld);wRecTrain(dsg.idx.vld)]');
            dsVld1= combine(dsX1vld_1,dsX1vld_3,dsX1vld_4,dsX1vld_6,dsX1vld_7,dsT1vld_1);

            if strcmp(TransferLearning,'True')
                NNs{i_}.inp2.trn = {inp2.DATABASE_1(:,idx2.trn)',inp2.DATABASE_3(:,idx2.trn)',inp2.DATABASE_4(:,idx2.trn)',inp2.DATABASE_6(:,idx2.trn)',inp2.DATABASE_7(:,idx2.trn)'};
                NNs{i_}.tar2.trn = {tar2.DATABASE_1(:,idx2.trn)'};
                NNs{i_}.inp2.vld = {inp2.DATABASE_1(:,idx2.vld)',inp2.DATABASE_3(:,idx2.vld)',inp2.DATABASE_4(:,idx2.vld)',inp2.DATABASE_6(:,idx2.vld)',inp2.DATABASE_7(:,idx2.vld)'};
                NNs{i_}.tar2.vld = {tar2.DATABASE_1(:,idx2.vld)'};
                NNs{i_}.inp2.tst = {inp2.DATABASE_1(:,idx2.tst)',inp2.DATABASE_3(:,idx2.tst)',inp2.DATABASE_4(:,idx2.tst)',inp2.DATABASE_6(:,idx2.tst)',inp2.DATABASE_7(:,idx2.tst)'};
                NNs{i_}.tar2.tst = {tar2.DATABASE_1(:,idx2.tst)'};


                dsX2Trn_1 = arrayDatastore(inp2.DATABASE_1(:,idx2.trn)');
                dsX2Trn_3 = arrayDatastore(inp2.DATABASE_3(:,idx2.trn)');
                dsX2Trn_4 = arrayDatastore(inp2.DATABASE_4(:,idx2.trn)');
                dsX2Trn_6 = arrayDatastore(inp2.DATABASE_6(:,idx2.trn)');
                dsX2Trn_7 = arrayDatastore(inp2.DATABASE_7(:,idx2.trn)');
                dsT2Trn_1 = arrayDatastore(tar2.DATABASE_1(:,idx2.trn)');
                dsTrn2 = combine(dsX2Trn_1,dsX2Trn_3,dsX2Trn_4,dsX2Trn_6,dsX2Trn_7,dsT2Trn_1);

                dsX2vld_1 = arrayDatastore(inp2.DATABASE_1(:,idx2.vld)');
                dsX2vld_3 = arrayDatastore(inp2.DATABASE_3(:,idx2.vld)');
                dsX2vld_4 = arrayDatastore(inp2.DATABASE_4(:,idx2.vld)');
                dsX2vld_6 = arrayDatastore(inp2.DATABASE_6(:,idx2.vld)');
                dsX2vld_7 = arrayDatastore(inp2.DATABASE_7(:,idx2.vld)');
                dsT2vld_1 = arrayDatastore(tar2.DATABASE_1(:,idx2.vld)');
                dsVld2= combine(dsX2vld_1,dsX2vld_3,dsX2vld_4,dsX2vld_6,dsX2vld_7,dsT2vld_1);
            end

        elseif index_extra>0 && n_fm>0 && n_rg>0
            NNs{i_}.inp.trn = {inp.DATABASE_1(:,dsg.idx.trn)',inp.DATABASE_3(:,dsg.idx.trn)',inp.DATABASE_6(:,dsg.idx.trn)',inp.DATABASE_7(:,dsg.idx.trn)'};
            NNs{i_}.tar.trn = {tar.DATABASE_1(:,dsg.idx.trn)'};
            NNs{i_}.inp.vld = {inp.DATABASE_1(:,dsg.idx.vld)',inp.DATABASE_3(:,dsg.idx.vld)',inp.DATABASE_6(:,dsg.idx.vld)',inp.DATABASE_7(:,dsg.idx.vld)'};
            NNs{i_}.tar.vld = {tar.DATABASE_1(:,dsg.idx.vld)'};
            NNs{i_}.inp.tst = {inp.DATABASE_1(:,dsg.idx.tst)',inp.DATABASE_3(:,dsg.idx.tst)',inp.DATABASE_6(:,dsg.idx.tst)',inp.DATABASE_7(:,dsg.idx.tst)'};
            NNs{i_}.tar.tst = {tar.DATABASE_1(:,dsg.idx.tst)'};

            dsX1Trn_1 = arrayDatastore(inp.DATABASE_1(:,dsg.idx.trn)');
            dsX1Trn_3 = arrayDatastore(inp.DATABASE_3(:,dsg.idx.trn)');
            dsX1Trn_6 = arrayDatastore(inp.DATABASE_6(:,dsg.idx.trn)');
            dsX1Trn_7 = arrayDatastore(inp.DATABASE_7(:,dsg.idx.trn)');
            dsT1Trn_1 = arrayDatastore(tar.DATABASE_1(:,dsg.idx.trn)');
            dsTrn1 = combine(dsX1Trn_1,dsX1Trn_3,dsX1Trn_6,dsX1Trn_7,dsT1Trn_1);

            dsX1vld_1 = arrayDatastore(inp.DATABASE_1(:,dsg.idx.vld)');
            dsX1vld_3 = arrayDatastore(inp.DATABASE_3(:,dsg.idx.vld)');
            dsX1vld_6 = arrayDatastore(inp.DATABASE_6(:,dsg.idx.vld)');
            dsX1vld_7 = arrayDatastore(inp.DATABASE_7(:,dsg.idx.vld)');
            dsT1vld_1 = arrayDatastore(tar.DATABASE_1(:,dsg.idx.vld)');
            dsVld1= combine(dsX1vld_1,dsX1vld_3,dsX1vld_6,dsX1vld_7,dsT1vld_1);

            if strcmp(TransferLearning,'True')
                NNs{i_}.inp2.trn = {inp2.DATABASE_1(:,idx2.trn)',inp2.DATABASE_3(:,idx2.trn)',inp2.DATABASE_6(:,idx2.trn)',inp2.DATABASE_7(:,idx2.trn)'};
                NNs{i_}.tar2.trn = {tar2.DATABASE_1(:,idx2.trn)'};
                NNs{i_}.inp2.vld = {inp2.DATABASE_1(:,idx2.vld)',inp2.DATABASE_3(:,idx2.vld)',inp2.DATABASE_6(:,idx2.vld)',inp2.DATABASE_7(:,idx2.vld)'};
                NNs{i_}.tar2.vld = {tar2.DATABASE_1(:,idx2.vld)'};
                NNs{i_}.inp2.tst = {inp2.DATABASE_1(:,idx2.tst)',inp2.DATABASE_3(:,idx2.tst)',inp2.DATABASE_6(:,idx2.tst)',inp2.DATABASE_7(:,idx2.tst)'};
                NNs{i_}.tar2.tst = {tar2.DATABASE_1(:,idx2.tst)'};


                dsX2Trn_1 = arrayDatastore(inp2.DATABASE_1(:,idx2.trn)');
                dsX2Trn_3 = arrayDatastore(inp2.DATABASE_3(:,idx2.trn)');
                dsX2Trn_6 = arrayDatastore(inp2.DATABASE_6(:,idx2.trn)');
                dsX2Trn_7 = arrayDatastore(inp2.DATABASE_7(:,idx2.trn)');
                dsT2Trn_1 = arrayDatastore(tar2.DATABASE_1(:,idx2.trn)');
                dsTrn2 = combine(dsX2Trn_1,dsX2Trn_3,dsX2Trn_6,dsX2Trn_7,dsT2Trn_1);

                dsX2vld_1 = arrayDatastore(inp2.DATABASE_1(:,idx2.vld)');
                dsX2vld_3 = arrayDatastore(inp2.DATABASE_3(:,idx2.vld)');
                dsX2vld_6 = arrayDatastore(inp2.DATABASE_6(:,idx2.vld)');
                dsX2vld_7 = arrayDatastore(inp2.DATABASE_7(:,idx2.vld)');
                dsT2vld_1 = arrayDatastore(tar2.DATABASE_1(:,idx2.vld)');
                dsVld2= combine(dsX2vld_1,dsX2vld_3,dsX2vld_6,dsX2vld_7,dsT2vld_1);
            end

        elseif index_extra>0 && n_classes>0
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