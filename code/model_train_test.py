# 引入库文件
import pandas as pd
import numpy as np
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping

from keras.layers import Input, Conv1D, GlobalAvgPool2D, GlobalAveragePooling1D, \
    Dropout, Dense, Activation, Concatenate, Multiply, MaxPool2D, Add,  \
    LSTM, Bidirectional, Conv2D, AveragePooling2D, BatchNormalization, Flatten, GlobalAveragePooling2D, \
    GlobalMaxPooling2D, Reshape, Permute, multiply, Lambda, add, subtract, MaxPooling2D, GRU, ReLU,GlobalMaxPooling1D
from keras.regularizers import l1, l2
from sklearn.metrics import roc_auc_score, roc_curve
from keras.models import Model, load_model
import tensorflow as tf
from keras import backend as K
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from keras.layers import Layer, MaxPooling1D, GaussianNoise, Embedding, AveragePooling1D
from keras import initializers
from random import randint, choice
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.use('TkAgg')
from Bio import SeqIO
from sklearn.metrics import precision_recall_curve, auc
# 忽略提醒
import warnings

warnings.filterwarnings("ignore")


# 读取DNA序列
def read_fasta(file_path):
    '''File_path: Path to the fasta file
       Returns: List of sequence
    '''
    one=list(SeqIO.parse(file_path,'fasta'))
    return one


def onehot(seq):
    bases = ['A','C','D','E','F','G','H','I','O','U','K','L','M','N','P','Q','R','S','T','V','W','Y']
    X = np.zeros((len(seq),len(seq[0]),len(bases)))
    for i,m in enumerate(seq):
        for l,s in enumerate(m):
            if s in bases:
                X[i,l,bases.index(s)] = 1
    return X

def to_embedding_numeric(seqs):
    numeric_sequences = []
    base_to_index = {'A': 0, 'C': 1, 'D': 2, 'E': 3,
                     'F': 4, 'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9, 'M': 10,
                     'N': 11, 'P': 12, 'Q': 13, 'R': 14, 'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19, 'O':20, 'U':21
                     }
    for dna_sequence in seqs:
        numeric_sequence = []
        for base in dna_sequence:
            numeric_sequence.append(base_to_index[base])
        numeric_sequences.append(numeric_sequence)
    return numeric_sequences

# 性能评价指标
def show_performance(y_true, y_pred):
    # 定义tp, fp, tn, fn初始值
    TP, FP, FN, TN = 0, 0, 0, 0

    for i in range(len(y_true)):
        if y_true[i] == 1:
            if y_pred[i] > 0.5:
                TP += 1
            else:
                FN += 1
        if y_true[i] == 0:
            if y_pred[i] > 0.5:
                FP += 1
            else:
                TN += 1

    # 计算敏感性Sn
    Sn = TP / (TP + FN + 1e-06)
    # 计算特异性Sp
    Sp = TN / (FP + TN + 1e-06)
    # 计算Acc值
    Acc = (TP + TN) / len(y_true)
    # 计算MCC：马修斯相关系数是在混淆矩阵环境中建立二元分类器预测质量的最具信息性的单一分数
    MCC = ((TP * TN) - (FP * FN)) / np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN) + 1e-06)

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)

    F1_score = 2 * (precision * recall) / (precision + recall)

    return Sn, Sp, Acc, MCC, F1_score

def performance_mean(performance):
    print('Sn = %.4f ± %.4f' % (np.mean(performance[:, 0]), np.std(performance[:, 0])))
    print('Sp = %.4f ± %.4f' % (np.mean(performance[:, 1]), np.std(performance[:, 1])))
    print('Acc = %.4f ± %.4f' % (np.mean(performance[:, 2]), np.std(performance[:, 2])))
    print('Mcc = %.4f ± %.4f' % (np.mean(performance[:, 3]), np.std(performance[:, 3])))
    print('Auc = %.4f ± %.4f' % (np.mean(performance[:, 4]), np.std(performance[:, 4])))
    print('F1_score = %.4f ± %.4f' % (np.mean(performance[:, 5]), np.std(performance[:, 5])))
    print('pr_auc = %.4f ± %.4f' % (np.mean(performance[:, 6]), np.std(performance[:, 6])))


# 残差块
def ResBlock(x, filters, kernel_size, dilation_rate):
    r = Conv1D(filters, kernel_size, padding='same', dilation_rate=dilation_rate, activation='relu')(x)  # 第一卷积
    r = Conv1D(filters, kernel_size, padding='same', dilation_rate=dilation_rate)(r)  # 第二卷积
    if x.shape[-1] == filters:
        shortcut = x
    else:
        shortcut = Conv1D(filters, kernel_size, padding='same')(x)  # shortcut（捷径）
    o = add([r, shortcut])
    o = Activation('relu')(o)  # 激活函数
    return o


def build_model(windows=29, weight_decay=1e-4):
    input_1 = Input(shape=(windows,))

    embedding = Embedding(20, 200, input_length=29)

    x_embedding = embedding(input_1)

    print(x_embedding.shape)

    x_1 = ResBlock(x_embedding, filters=32, kernel_size=3, dilation_rate=1)
    x_1 = ResBlock(x_1, filters=32, kernel_size=3, dilation_rate=2)

    x_2 = Bidirectional(GRU(200, return_sequences=True))(x_embedding)
    x_2 = Bidirectional(GRU(200, return_sequences=True))(x_2)



    # # Flatten
    x_1 = Flatten()(x_1)
    x_2 = Flatten()(x_2)

    # MLP
    x_1 = Dense(units=240, activation="sigmoid", use_bias=False,
              kernel_initializer='he_normal',
              kernel_regularizer=l2(weight_decay))(x_1)

    x_1 = Dropout(0.5)(x_1)

    x_2 = Dense(units=240, activation="sigmoid", use_bias=False,
                kernel_initializer='he_normal',
                kernel_regularizer=l2(weight_decay))(x_2)

    x_2 = Dropout(0.5)(x_2)
    #
    x = add([x_1, x_2])

    #
    x = BatchNormalization()(x)

    x = Dense(units=2, activation="softmax", use_bias=False,
              kernel_initializer='he_normal',
              kernel_regularizer=l2(weight_decay))(x)


    inputs = [input_1]
    outputs = [x]

    model = Model(inputs=inputs, outputs=outputs, name="Kcr")

    optimizer = Adam(learning_rate=0.5e-4, epsilon=1e-8)

    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model


if __name__ == '__main__':

    np.random.seed(0)
    tf.random.set_seed(1)  # for reproducibility

    # step1：读取数据
    # 读取训练集
    # Read the training set
    train_pos_seqs = np.array(read_fasta('../dataset/train_pos.txt'))
    train_neg_seqs = np.array(read_fasta('../dataset/train_neg.txt'))

    train_seqs = np.concatenate((train_pos_seqs, train_neg_seqs), axis=0)


    train = np.array(to_embedding_numeric(train_seqs)).astype(np.float32)
    print('train', train.shape)

    train_label = np.array([1] * 12262 + [0] * 60101).astype(np.float32)
    train_label = to_categorical(train_label, num_classes=2)

    # Read the testing set
    test_pos_seqs = np.array(read_fasta('../dataset/test_pos.txt'))
    test_neg_seqs = np.array(read_fasta('../dataset/test_neg.txt'))

    test_seqs = np.concatenate((test_pos_seqs, test_neg_seqs), axis=0)

    test = np.array(to_embedding_numeric(test_seqs)).astype(np.float32)

    test_label = np.array([1] * 3343 + [0] * 15010).astype(np.float32)
    test_label = to_categorical(test_label, num_classes=2)


    # 构建模型
    model = build_model()

    # 超参数设置 120
    BATCH_SIZE = 120
    EPOCHS = 200
    weights = {0: 1, 1: 4.9}

    # Cross-validation
    n = 5
    k_fold = KFold(n_splits=n, shuffle=True, random_state=42)

    all_performance = []
    tprs = []
    mean_fpr = np.linspace(0, 1, 100)
    for fold_count, (train_index, val_index) in enumerate(k_fold.split(train)):
        print('*' * 30 + ' the ' + str(fold_count + 1) + ' fold ' + '*' * 30)

        trains, val = train[train_index], train[val_index]
        trains_label, val_label = train_label[train_index], train_label[val_index]

        model.fit(x=trains, y=trains_label, validation_data=(val, val_label), epochs=EPOCHS,
                          batch_size=BATCH_SIZE, shuffle=True,class_weight=weights,
                          callbacks=[EarlyStopping(monitor='val_loss', patience=13, mode='auto')],
                          verbose=1)

        # # 保存模型

        model.save('../models/model_fold' + str(fold_count+1) + '.h5')

        del model

        model = load_model('../models/model_fold' + str(fold_count+1) + '.h5')

        val_pred = model.predict(val, verbose=1)

        # Sn, Sp, Acc, MCC, AUC, F1_score, pr_auc
        Sn, Sp, Acc, MCC, F1_score = show_performance(val_label[:, 1], val_pred[:, 1])
        AUC = roc_auc_score(val_label[:, 1], val_pred[:, 1])
        precision, recall, thresholds = precision_recall_curve(val_label[:, 1], val_pred[:, 1])
        pr_auc = auc(recall, precision)
        print('-----------------------------------------------val---------------------------------------')
        print('Sn = %f, Sp = %f, Acc = %f, MCC = %f, AUC = %f, F1_score = %f, pr_auc = %f' % (
        Sn, Sp, Acc, MCC, AUC, F1_score, pr_auc))

        val_performance = [Sn, Sp, Acc, MCC, AUC, F1_score, pr_auc]
        all_performance.append(val_performance)


        '''Mapping the ROC'''
        fpr, tpr, thresholds = roc_curve(val_label[:, 1], val_pred[:, 1], pos_label=1)
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        plt.plot(fpr, tpr, label='ROC fold {} (AUC={:.4f})'.format(str(fold_count + 1), AUC))


    all_performance = np.array(all_performance)
    print('5 fold result:', all_performance)
    performance_mean = performance_mean(all_performance)

    '''Mapping the ROC'''
    plt.plot([0, 1], [0, 1], '--', color='red')
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = np.mean(np.array(all_performance)[:, 4])
    plt.plot(mean_fpr, mean_tpr, color='b', label=r'Mean ROC (AUC=%0.4f)' % (mean_auc), lw=2, alpha=.8)
    plt.title('ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.savefig('../images/5fold_ROC_Curve.jpg', dpi=1200, bbox_inches='tight')
    plt.legend(loc='lower right')
    plt.show()


    model.fit(x=train, y=train_label, validation_data=(test, test_label), epochs=EPOCHS,
                      batch_size=BATCH_SIZE, shuffle=True, class_weight=weights,
                      callbacks=[EarlyStopping(monitor='val_loss', patience=20, mode='auto')],
                      verbose=1)
    # 保存模型

    model.save('../models/model_test.h5')

    del model

    model = load_model('../models/model_test.h5')

    test_score = model.predict(test)

    # Sn, Sp, Acc, MCC, AUC, F1_score, pr_auc
    Sn, Sp, Acc, MCC, F1_score = show_performance(test_label[:, 1], test_score[:, 1])
    AUC = roc_auc_score(test_label[:, 1], test_score[:, 1])

    precision, recall, thresholds = precision_recall_curve(test_label[:, 1], test_score[:, 1])
    pr_auc = auc(recall, precision)
    print('-----------------------------------------------test---------------------------------------')
    print('Sn = %f, Sp = %f, Acc = %f, MCC = %f, AUC = %f, F1_score = %f, pr_auc = %f' % (
    Sn, Sp, Acc, MCC, AUC, F1_score, pr_auc))


    '''Mapping the ROC'''
    plt.plot([0, 1], [0, 1], '--', color='red')
    test_fpr, test_tpr, thresholds = roc_curve(test_label[:,1], test_score[:,1], pos_label=1)

    plt.plot(test_fpr, test_tpr, color='b', label=r'test ROC (AUC=%0.4f)' % (AUC), lw=2, alpha=.8)
    plt.title('ROC Curve OF')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.savefig('../images/test_ROC_Curve.jpg', dpi=1200, bbox_inches='tight')
    plt.legend(loc='lower right')
    plt.show()




