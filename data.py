# -*- coding: utf-8 -*-

from collections import defaultdict
import torch
import pandas as pd

def loadlncseq(trainFile, splitMark):
    print(trainFile)
    lnc_seq = pd.DataFrame(columns=["L1", "L2", "L3", "L4", "L5", "L6", "L7", "L8", "L9",
                                    "L10", "L11", "L12", "L13", "L14", "L15", "L16", "L17", "L18", "L19",
                                    "L20", "L21", "L22", "L23", "L24", "L25", "L26", "L27", "L28", "L29",
                                    "L30", "L31", "L32", "L33", "L34", "L35", "L36", "L37", "L38", "L39",
                                    "L40", "L41", "L42", "L43", "L44", "L45", "L46", "L47", "L48", "L49",
                                    "L50", "L51", "L52", "L53", "L54", "L55", "L56", "L57", "L58", "L59",
                                    "L60", "L61", "L62", "L63", "L64"])

    index = 0

    for line in open(trainFile):

        L1, L2, L3, L4, L5, L6, L7, L8, L9,\
        L10, L11, L12, L13, L14, L15, L16, L17, L18, L19,\
        L20, L21, L22, L23, L24, L25, L26, L27, L28, L29,\
        L30, L31, L32, L33, L34, L35, L36, L37, L38, L39,\
        L40, L41, L42, L43, L44, L45, L46, L47, L48, L49,\
        L50, L51, L52, L53, L54, L55, L56, L57, L58, L59,\
        L60, L61, L62, L63, L64 = line.strip().split(splitMark)
       
        lnc_seq.loc['%d' % index] = [L1, L2, L3, L4, L5, L6, L7, L8, L9,\
                                    L10, L11, L12, L13, L14, L15, L16, L17, L18, L19,\
                                    L20, L21, L22, L23, L24, L25, L26, L27, L28, L29,\
                                    L30, L31, L32, L33, L34, L35, L36, L37, L38, L39,\
                                    L40, L41, L42, L43, L44, L45, L46, L47, L48, L49,\
                                    L50, L51, L52, L53, L54, L55, L56, L57, L58, L59,\
                                    L60, L61, L62, L63, L64]

        index = index + 1
        lnc_seq.to_csv("lncRNA_embedding_save.csv", index=False)

    return lnc_seq

# 返回训练集合
def loadTrainingData(trainFile,splitMark):
    trainSet = defaultdict(list)
    max_l_id = -1
    max_d_id = -1
    for line in open(trainFile):
        l_Id, d_Id = line.strip().split(splitMark)

        l_Id = int(l_Id)  #用户ID变为整型 943
        d_Id = int(d_Id)  #项目ID变为整型 1330
        trainSet[l_Id].append(d_Id)
        max_l_id = max(l_Id, max_l_id)
        max_d_id = max(d_Id, max_d_id)

    lncCount = max_l_id + 1
    disCount = max_d_id + 1
    print("Training data loading done")

    return trainSet, lncCount, disCount

# 返回测试集
def loadTestData(testFile,splitMark):
    testSet = defaultdict(list)
    max_l_id = -1
    max_d_id = -1
    for line in open(testFile):
        l_Id, d_Id = line.strip().split(splitMark)
        l_Id = int(l_Id)
        d_Id = int(d_Id)
        testSet[l_Id].append(d_Id)
        max_l_id = max(l_Id, max_l_id)
        max_d_id = max(d_Id, max_d_id)
    lncCount = max_l_id + 1
    disCount = max_d_id + 1
    print("Test data loading done")

    return testSet, lncCount, disCount

# 返回训练集向量与测试集向量
def to_Vectors(trainSet, lncCount, disCount, lnc_List_test, mode):
    
    testMaskDict = defaultdict(lambda: [0] * disCount)
    batchCount = lncCount
    if mode == "disBased":
        lncCount = disCount
        disCount = batchCount
        batchCount = lncCount
    trainDict = defaultdict(lambda: [0] * disCount)
    for l_Id, i_list in trainSet.items():
        for d_Id in i_list:
            testMaskDict[l_Id][d_Id] = -99999
            if mode == "lncBased":
                trainDict[l_Id][d_Id] = 1.0
            else:
                trainDict[d_Id][l_Id] = 1.0

    trainVector = []

    for batchId in range(batchCount):
        trainVector.append(trainDict[batchId])

    testMaskVector = []
    for l_Id in lnc_List_test:
        testMaskVector.append(testMaskDict[l_Id])

    return (torch.Tensor(trainVector)), torch.Tensor(testMaskVector), batchCount

