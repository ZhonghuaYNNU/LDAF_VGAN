import random
import copy
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.autograd import Variable
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve, auc
from sklearn.metrics import roc_curve
import metrics

import data
import bayes_gan


def select_negative_items(realData, num_pm, num_zr, discount):
    data = np.array(realData)
    n_dis_pm = np.zeros_like(data)
    n_dis_zr = np.zeros_like(data)
    all_dis_index = []
    for i in range(data.shape[0]):
        p_dis = np.where(data[i] != 0)[0]
        all_dis_index_1 = random.sample(range(data.shape[1]), discount)

        for j in all_dis_index_1:
            if j not in p_dis:
                all_dis_index.append(j)

        random.shuffle(all_dis_index)  # 将列表中的元素打乱顺序
        n_dis_index_pm = all_dis_index[0: num_pm]
        n_dis_index_zr = all_dis_index[num_pm: (num_pm + num_zr)]
        n_dis_pm[i][n_dis_index_pm] = 1
        n_dis_zr[i][n_dis_index_zr] = 1
    return n_dis_pm, n_dis_zr


def computeTopN(groundTruth, result, topN):
    result = result.tolist()  #将result转为列表

    for i in range(len(result)):
        result[i] = (result[i], i)
    result.sort(key=lambda x: x[0], reverse=True)
    hit = 0
    for i in range(topN):
        if(str(result[i][1]) in str(groundTruth)):
            hit = hit + 1
    return hit/topN



def main(lncCount, disCount, testSet, trainVector, trainMaskVector, \
         lncseq_pre, topN, epochCount, pro_ZR, pro_PM, alpha):

    lnc_seq_shape = lncseq_pre.shape[1]

    lncseq_pre = lncseq_pre.values  # 取出表里的值

    lncseq_pre = np.insert(lncseq_pre, 0, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                           0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                           0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                           0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                           0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                           0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                           0, 0, 0, 0], axis=0)

    lncseq_pre = torch.tensor(lncseq_pre.astype(np.float32))  # 把表格数据转为张量
    result_precision = np.zeros((1, 2))  # 结果为[[0,0]]

    # Build the generator and discriminator
    G = bayes_gan.generator(disCount)
    D = bayes_gan.discriminator(disCount)
    regularization = nn.MSELoss()
    d_optimizer = torch.optim.Adam(D.parameters(), lr=0.0002)
    g_optimizer = torch.optim.Adam(G.parameters(), lr=0.0002)
    G_step = 3  #原值为5
    D_step = 1  #原值为2
    batchSize_G = 32
    batchSize_D = 32

    criterion = metrics.ELBO(len(trainSet))

    result_auc_sum = 0
    noise = np.random.uniform(0, 1, size=[lncRNACount, diseaseCount])
    noise = torch.Tensor(noise)
    for epoch in range(epochCount):

        # ---------------------
        #  Train Generator
        # ---------------------

        for step in range(G_step):
            # Select a random batch of purchased vector
            leftIndex = random.randint(1, lncCount - batchSize_G - 1)
            realData = Variable(copy.deepcopy(trainVector[leftIndex:leftIndex + batchSize_G]))
            eu = Variable(copy.deepcopy(trainVector[leftIndex:leftIndex + batchSize_G]))
            lncseq_batch = Variable(copy.deepcopy(lncseq_pre[leftIndex:leftIndex + batchSize_G]))
            noise_G = Variable(copy.deepcopy(noise[leftIndex:leftIndex + batchSize_G]))

            n_dis_pm, n_dis_zr = select_negative_items(realData, pro_PM, pro_ZR, disCount)

            ku_zp = Variable(torch.tensor(n_dis_pm + n_dis_zr))
            realData_zp = Variable(torch.ones_like(realData)) * eu + Variable(torch.zeros_like(realData)) * ku_zp

            fakeData, g_kl = G(noise_G)
            beta = metrics.get_beta(step - 1, len(trainSet), 0.1, epoch, epochCount)

            fakeData_ZP = fakeData * (eu + ku_zp)
            fakeData_result, fakeData_result_kl = D(fakeData_ZP)

            # Train the discriminator
            g_loss = (np.mean(np.log(1. - fakeData_result.detach().numpy() + 10e-5)) * len(trainSet) + alpha * regularization(
                     fakeData_ZP, realData_zp) * len(trainSet)) + beta * fakeData_result_kl

            g_optimizer.zero_grad()
            g_loss.backward(retain_graph=True)
            g_optimizer.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------
        for step in range(D_step):
            # Select a random batch of purchased vector
            leftIndex = random.randint(1, lncCount - batchSize_D - 1)
            realData = Variable(copy.deepcopy(trainVector[leftIndex:leftIndex + batchSize_D]))
            eu = Variable(copy.deepcopy(trainVector[leftIndex:leftIndex + batchSize_D]))
            lncseq_batch = Variable(copy.deepcopy(lncseq_pre[leftIndex:leftIndex + batchSize_D]))
            noise_D = Variable(copy.deepcopy(noise[leftIndex:leftIndex + batchSize_D]))

            # Select a random batch of negative items for every user
            n_dis_pm, _ = select_negative_items(realData, pro_PM, pro_ZR, disCount)
            ku = Variable(torch.tensor(n_dis_pm))
            realData_zp = Variable(torch.ones_like(realData)) * eu + Variable(torch.zeros_like(realData)) * ku

            # Generate a batch of new purchased vector

            fakeData, _ = G(noise_D)
            beta = metrics.get_beta(step - 1, len(trainSet), 0.1, epoch, epochCount)
            fakeData_ZP = fakeData * (eu + ku)

            # Train the discriminator
            fakeData_result, d_fake_kl = D(fakeData_ZP)

            # 鉴别器输入真实数据
            realData_result, d_real_kl = D(realData)
            
            d_loss = (-np.mean(np.log(realData_result.detach().numpy() + 10e-5) +
                    np.log(1. - fakeData_result.detach().numpy() + 10e-5))) * len(trainSet) + beta * (d_fake_kl+d_real_kl)\
                    + alpha * regularization(fakeData_ZP, realData_zp)

            d_optimizer.zero_grad()
            d_loss.backward(retain_graph=True)
            d_optimizer.step()

        if (epoch % 20 == 0):
            n_lnc = len(testSet)
            index = 0
            precisions = 0
            auc_result_G_all = []
            label = []
            pred = []
            for testlnc in testSet.keys():

                data = Variable(copy.deepcopy(noise[testlnc]))

                #  Exclude the purchased vector that have occurred in the training
                result_G = G(data.reshape(1, disCount))
                result1 = result_G[0].reshape(disCount).detach().numpy()
                result2 = result_G[0] + Variable(copy.deepcopy(trainMaskVector[index]))
                result3 = result2.reshape(disCount)

                test_i = testSet[testlnc]
                test = [0] * disCount
                for value in test_i:
                    test[value] = 1
                pred_i = list(result1)
                train_i = trainVector[testlnc].tolist()

                for i in range(disCount - 1, -1, -1):
                    if train_i[i] == 1:
                        del test[i]
                        del pred_i[i]

                label += test
                pred += pred_i
                precision = computeTopN(testSet[testlnc], result3, topN)
                precisions += precision
                index += 1

            auc1 = roc_auc_score(label, pred)

            auc_result_G_all.append(auc1)
            precisions = precisions / n_lnc
            result_precision = np.concatenate((result_precision, np.array([[epoch, precisions]])), axis=0)

            fpr, tpr, thresholds = roc_curve(label, pred)
            aupr_precision, aupr_recall, aucr_thresholds = precision_recall_curve(label, pred)
            aupr = auc(aupr_recall,aupr_precision)
            np.savetxt("main_bayes_fpr_3762", fpr, "%f")
            np.savetxt("main_bayes_tpr_3762", tpr, "%f")
            print('Epoch[{}/{}],d_loss:{:.6f},g_loss:{:.6f},precision:{},auc:{},aupr:{}'.format(epoch, epochCount,
                                                                                 d_loss.item(),
                                                                                 g_loss.item(),
                                                                                 precisions,auc1,aupr))

    return result_precision

if __name__ == '__main__':
    topN = 10
    epochs = 3000
    pro_ZR = 50  #原值为50
    pro_PM = 50  #原值为50
    alpha = 0.8

    lnc_seq = data.loadlncseq("lncrna_seq_feat_3kmer", ",")

    trainSet, train_lncRNA, train_disease = data.loadTrainingData("train1", " ")

    testSet, test_lncRNA, test_disease = data.loadTestData("test1", " ")

    lncRNACount = max(train_lncRNA, test_lncRNA)
    diseaseCount = max(train_disease, test_disease)
    disease_List_test = list(testSet.keys())
    # print(lncRNACount)
    # print(diseaseCount)

    trainVector, trainMaskVector, batchCount = data.to_Vectors(trainSet, lncRNACount,\
                                                               diseaseCount, disease_List_test, "lncBased")

    result_precision = main(lncRNACount, diseaseCount, testSet,\
                            trainVector, trainMaskVector, lnc_seq, topN, epochs, pro_ZR, pro_PM, alpha)

    result_precision = result_precision[1:, ]



