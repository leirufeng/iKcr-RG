
import numpy as np
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping

from keras.layers import Input, Conv1D, GlobalAvgPool2D, GlobalAveragePooling1D, \
    Dropout, Dense, Activation, Concatenate, Multiply, MaxPool2D, Add,  \
    LSTM, Bidirectional, Conv2D, AveragePooling2D, BatchNormalization, Flatten, GlobalAveragePooling2D, \
    GlobalMaxPooling2D, Reshape, Permute, multiply, Lambda, add, subtract, MaxPooling2D, GRU, ReLU
from keras.regularizers import l1, l2
from sklearn.metrics import roc_auc_score, roc_curve
from keras.models import Model, load_model
import tensorflow as tf
from keras import backend as K
from keras.optimizer_v2.adam import Adam
from sklearn.model_selection import KFold
from keras.layers import Embedding

import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.use('TkAgg')
from Bio import SeqIO

import warnings

warnings.filterwarnings("ignore")

# read DNA sequences
def read_fasta(file_path):
    '''File_path: Path to the fasta file
       Returns: List of sequence
    '''
    one=list(SeqIO.parse(file_path,'fasta'))
    return one


def protein_to_Kmer(seqs,K):
    numeric_sequences = []
    base_to_index = {'A': 0, 'C': 1, 'D': 2, 'E': 3,
                     'F': 4, 'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9, 'M': 10,
                     'N': 11, 'P': 12, 'Q': 13, 'R': 14, 'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19
                     }
    for dna_sequence in seqs:
        numeric_sequence = []
        for base in dna_sequence:
            numeric_sequence.append(base_to_index[base])

        kmer_sequence = []
        for i in range(len(numeric_sequence) - (K - 1)):
            kmer = numeric_sequence[i:i + K]
            kmer_marge = [int("".join(map(str, kmer)))]
            kmer_sequence.append(kmer_marge)

        numeric_sequences.append(kmer_sequence)
    return numeric_sequences


# Performance evaluation
def show_performance(y_true, y_pred):

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


    Sn = TP / (TP + FN + 1e-06)

    Sp = TN / (FP + TN + 1e-06)

    Acc = (TP + TN) / len(y_true)

    MCC = ((TP * TN) - (FP * FN)) / np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN) + 1e-06)

    return Sn, Sp, Acc, MCC

def performance_mean(performance):
    print('Sn = %.4f ± %.4f' % (np.mean(performance[:, 0]), np.std(performance[:, 0])))
    print('Sp = %.4f ± %.4f' % (np.mean(performance[:, 1]), np.std(performance[:, 1])))
    print('Acc = %.4f ± %.4f' % (np.mean(performance[:, 2]), np.std(performance[:, 2])))
    print('Mcc = %.4f ± %.4f' % (np.mean(performance[:, 3]), np.std(performance[:, 3])))
    print('Auc = %.4f ± %.4f' % (np.mean(performance[:, 4]), np.std(performance[:, 4])))

# Define a single convolutional layer in a Dense block
def conv_factory(x, filters, dropout_rate, weight_decay=1e-4):
    x = Activation('elu')(x)
    x = Conv2D(filters=filters,
               kernel_size=(3, 3),
               kernel_initializer="he_normal",
               padding="same",
               use_bias=False,
               kernel_regularizer=l2(weight_decay))(x)
    x = Dropout(dropout_rate)(x)
    return x


# Defining the transition layer
def transition(x, filters, dropout_rate, weight_decay=1e-4):

    x = Conv2D(filters=filters,
               kernel_size=(1, 1),
               kernel_initializer="he_normal",
               padding="same",
               use_bias=False,
               kernel_regularizer=l2(weight_decay))(x)
    x = Dropout(dropout_rate)(x)
    x = AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = BatchNormalization(axis=-1)(x)
    return x


# Define Dense blocks
def denseblock(x, layers, filters, growth_rate, dropout_rate=None, weight_decay=1e-4):
    list_feature_map = [x]

    for i in range(layers):
        x = conv_factory(x, growth_rate,
                         dropout_rate, weight_decay)
        list_feature_map.append(x)
        x = Concatenate(axis=-1)(list_feature_map)
        filters = filters + growth_rate
    return x, filters


# Residual Channel Attention Module
def channel_attention(input_feature, ratio=8):
    channel = input_feature.shape[-1]

    shared_layer_one = Dense(channel // ratio,
                             activation='relu',
                             kernel_initializer='he_normal',
                             use_bias=True,
                             bias_initializer='zeros')
    shared_layer_two = Dense(channel,
                             kernel_initializer='he_normal',
                             use_bias=True,
                             bias_initializer='zeros')

    avg_pool = GlobalAveragePooling2D()(
        input_feature)
    avg_pool_1 = Reshape((1, 1, channel))(avg_pool)
    # assert avg_pool._keras_shape[1:] == (1,1,channel)
    avg_pool = shared_layer_one(avg_pool_1)
    # assert avg_pool._keras_shape[1:] == (1,1,channel//ratio)
    avg_pool = shared_layer_two(avg_pool)
    # assert avg_pool._keras_shape[1:] == (1,1,channel)

    max_pool = GlobalMaxPooling2D()(
        input_feature)
    max_pool_1 = Reshape((1, 1, channel))(max_pool)
    # assert max_pool._keras_shape[1:] == (1,1,channel)
    max_pool = shared_layer_one(max_pool_1)
    # assert max_pool._keras_shape[1:] == (1,1,channel//ratio)
    max_pool = shared_layer_two(max_pool)
    # assert max_pool._keras_shape[1:] == (1,1,channel)

    cbam_feature = Add()([avg_pool, max_pool])
    cbam_feature = Activation('sigmoid')(cbam_feature)

    # short-circuit connection
    cbam_feature_1 = Add()([avg_pool_1, max_pool_1])
    cbam_feature_1 = Activation('sigmoid')(cbam_feature_1)
    cbam_feature = Add()([cbam_feature, cbam_feature_1])

    return multiply([input_feature, cbam_feature])



# Building the model
def build_model(windows=31, denseblocks=4, layers=2, filters=96,
                growth_rate=32, dropout_rate=0.5, weight_decay=1e-4):
    input_1 = Input(shape=(windows,))

    # Word embedding coding
    embedding = Embedding(20, 80, input_length=31)

    x_1 = embedding(input_1)
    x_1 = K.expand_dims(x_1, -1)
    print(x_1.shape)

    for i in range(denseblocks - 1):
        # Add denseblock
        x_1, filters_1 = denseblock(x_1, layers=layers,
                                    filters=filters, growth_rate=growth_rate,
                                    dropout_rate=dropout_rate, weight_decay=weight_decay)
        # Add BatchNormalization
        x_1 = BatchNormalization(axis=-1)(x_1)

        # Add transition
        x_1 = transition(x_1, filters=filters_1,
                         dropout_rate=dropout_rate, weight_decay=weight_decay)
    # The last denseblock
    # Add denseblock
    x_1, filters_1 = denseblock(x_1, layers=layers,
                                filters=filters, growth_rate=growth_rate,
                                dropout_rate=dropout_rate, weight_decay=weight_decay)
    # Add BatchNormalization
    x_1 = BatchNormalization(axis=-1)(x_1)

    # # Adding an attention module layer
    x_1 = channel_attention(x_1)

    # # Flatten
    x = Flatten()(x_1)

    # MLP


    x = Dense(units=2, activation="softmax", use_bias=True,
              kernel_initializer='he_normal',
              kernel_regularizer=l2(weight_decay))(x)


    inputs = [input_1]
    outputs = [x]

    model = Model(inputs=inputs, outputs=outputs, name="Kcr")

    optimizer = Adam(learning_rate=1e-4, epsilon=1e-7)

    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model


if __name__ == '__main__':

    np.random.seed(0)
    tf.random.set_seed(1)  # for reproducibility


    # Read the training set
    train_pos_seqs = np.array(read_fasta('../data/Training_data/pos_fasta_lysine.txt'))
    train_neg_seqs = np.array(read_fasta('../data/Training_data/neg_fasta_lysine.txt'))

    train_seqs = np.concatenate( (train_pos_seqs, train_neg_seqs), axis=0 )

    train = np.array(protein_to_Kmer(train_seqs,1)).astype(np.float32)

    train_label = np.array( [1] * 6975 + [0] * 6975 ).astype( np.float32 )
    train_label = to_categorical( train_label, num_classes=2 )

    # Read the testing set
    test_pos_seqs = np.array(read_fasta('../data/Independent_data/pos_ind_testdata.txt'))
    test_neg_seqs = np.array(read_fasta('../data/Independent_data/neg_ind_testdata.txt'))

    test_seqs = np.concatenate((test_pos_seqs, test_neg_seqs), axis=0)

    test = np.array(protein_to_Kmer(test_seqs,1)).astype(np.float32)

    test_label = np.array([1] * 2989 + [0] * 2989).astype(np.float32)
    test_label = to_categorical(test_label, num_classes=2)

    # reading model

    model = build_model()


    BATCH_SIZE = 60
    EPOCHS = 200

    # # Cross-validation
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
                          batch_size=BATCH_SIZE, shuffle=True,
                          callbacks=[EarlyStopping(monitor='val_loss', patience=13, mode='auto')],
                          verbose=1)

        # 保存模型

        model.save('../models/model_fold' + str(fold_count+1) + '.h5')

        del model

        model = load_model('../models/model_fold' + str(fold_count+1) + '.h5')

        val_pred = model.predict(val, verbose=1)

        # Sn, Sp, Acc, MCC, AUC
        Sn, Sp, Acc, MCC = show_performance(val_label[:, 1], val_pred[:, 1])
        AUC = roc_auc_score(val_label[:, 1], val_pred[:, 1])
        print('Sn = %f, Sp = %f, Acc = %f, MCC = %f, AUC = %f' % (Sn, Sp, Acc, MCC, AUC))

        performance = [Sn, Sp, Acc, MCC, AUC]
        all_performance.append(performance)

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
                      batch_size=BATCH_SIZE, shuffle=True,
                      callbacks=[EarlyStopping(monitor='val_loss', patience=20, mode='auto')],
                      verbose=1)

    model.save('../models/model_test.h5')

    del model

    model = load_model('../models/model_test.h5')

    test_score = model.predict(test)


    # Sn, Sp, Acc, MCC, AUC
    Sn, Sp, Acc, MCC = show_performance(test_label[:,1], test_score[:,1])
    AUC = roc_auc_score(test_label[:,1], test_score[:,1])

    print('-----------------------------------------------test---------------------------------------')
    print('Sn = %f, Sp = %f, Acc = %f, MCC = %f, AUC = %f' % (Sn, Sp, Acc, MCC, AUC))

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


