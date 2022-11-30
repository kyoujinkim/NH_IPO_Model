import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import OneHotEncoder
from customclass.custommod import resDense, CategoricalAttention_v2, f1_m, CustomStopper
from matplotlib import font_manager, rc, use
from sklearn.metrics import confusion_matrix, f1_score
from joblib import load
import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def load_enc_transform(encname, arr):
    enc = load('./onehot_encoders/{}.joblib'.format(encname))
    onehotvector = enc.transform(np.array(arr).reshape(-1, 1)).toarray()

    return onehotvector

use("TkAgg")
#Korean font setting
font_path = 'C:/Windows/Fonts/gulim.ttc'
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)

def make_sourcearr(splitdate, endsplit='9999-12-31', file_path='./backdata/'):
    #import source data
    #encoder should be CP949 to display Korean
    bd_cnn = pd.read_csv(file_path+'cnnadj.csv', encoding='CP949', index_col='Code').T
    bd_cnn = bd_cnn.drop(labels=['Name','MKT Cap'], axis=1).astype('float')
    #clip outliers
    upperbound = pd.read_csv(file_path+'upperbound.csv', encoding='CP949', index_col=0)
    bd_cnn = bd_cnn.clip(lower=0, upper=upperbound.values.flatten(), axis=1)

    bd_EClstm = pd.read_csv(file_path+'econ.csv',
                           index_col=0, parse_dates=True,
                           encoding='CP949').fillna(value=0)

    bd_Slstm = pd.read_csv(file_path+'SWNEMO_Score.csv',
                           index_col=0, parse_dates=True,
                           encoding='CP949')
    bd_Slstm = bd_Slstm.rolling(13).mean().dropna()

    bd_bb = pd.read_csv(file_path+'BB.csv', encoding='CP949', index_col='종목코드')
    bd_bb.기업집단 = bd_bb.기업집단.apply(lambda x: 0 if x is np.nan else 1)

    match_comp = list(set(bd_cnn.index) & set(bd_bb.index))
    bd_cnn = bd_cnn.loc[match_comp]
    bd_bb = bd_bb.loc[match_comp]

    onehotE = OneHotEncoder()
    #make backdata
    bb_name = ['시총비중','신주비율','구주비율','유통비율','우리사주','기관투자자','일반투자자','상장주선수']
    labelarr = []
    bbarr = []
    cnnarr = []
    Plstmarr = []
    EClstmarr = []
    Slstmarr = []
    categarr = []
    montharr = []
    grouparr = []
    mktarr = []
    sipoarr = []
    resultarr = []
    datearr = [] #for reindexing
    for _, eachrow in bd_bb.iterrows():
        #if label is empty, skip number
        if pd.isna(eachrow['시가상승률']) or eachrow['시가상승률']==0:
            continue
        #define elements of company_idx
        compcode = eachrow.name
        groupnm = eachrow.기업집단
        bbdate = pd.to_datetime(eachrow['수요예측일'])

        #define datas
        temp_bb = eachrow[bb_name].values
        temp_plstm = bd_bb['시가상승률'][bd_bb['상장일']<eachrow['수요예측일']].iloc[:10].values
        temp_plstm = np.pad(temp_plstm, pad_width=(10-len(temp_plstm),0))
        temp_cnn = np.vstack([bd_cnn.loc[compcode].filter(regex='FY0').values,
                              bd_cnn.loc[compcode].filter(regex='FY-1').values,
                              bd_cnn.loc[compcode].filter(regex='FY-2').values]).T
        temp_EClstm = bd_EClstm.iloc[max(bd_EClstm.index.get_loc(bbdate, method='ffill') - 60,0):bd_EClstm.index.get_loc(bbdate,method='ffill')].values
        temp_Slstm = bd_Slstm.iloc[bd_Slstm.index.get_loc(bbdate, method='ffill') - 13:bd_Slstm.index.get_loc(bbdate,method='ffill')].values

        mktarr.append(eachrow['시장구분'])
        labelarr.append(compcode)
        bbarr.append(temp_bb)
        cnnarr.append(temp_cnn)
        Plstmarr.append(temp_plstm)
        EClstmarr.append(temp_EClstm)
        Slstmarr.append(temp_Slstm)
        categarr.append(eachrow['W_Sector'])
        montharr.append(pd.to_datetime(eachrow['수요예측일']).month)
        grouparr.append(groupnm)
        sipoarr.append(eachrow['특례상장'])
        resultarr.append(np.array([0,1])) if eachrow['시가상승률']>1.2 else resultarr.append(np.array([1,0]))
        datearr.append(eachrow['수요예측일'])

    labelarr = np.array(labelarr)
    bbarr = np.array(bbarr).astype('float')
    cnnarr = np.array(cnnarr).astype('float')
    Plstmarr = np.array(Plstmarr).astype('float')
    EClstmarr = np.array(EClstmarr).astype('float')
    Slstmarr = np.array(Slstmarr).astype('float')
    sipoarr = np.array(sipoarr).astype('float')

    mktarr = load_enc_transform('mktenc', mktarr)
    categarr = load_enc_transform('categenc', categarr)
    montharr = load_enc_transform('monthenc', montharr)
    grouparr = load_enc_transform('groupenc', grouparr)
    sipoarr = load_enc_transform('sipoenc', sipoarr)

    resultarr = np.array(resultarr).astype('float')

    datearr_df = pd.DataFrame(range(len(datearr)), index=datearr)
    trainidx = datearr_df[datearr_df.index < splitdate].values.flatten()
    testidx = datearr_df[(datearr_df.index >= splitdate) & (datearr_df.index < endsplit)].values.flatten()

    Slstmarr = np.reshape(Slstmarr, np.append(Slstmarr.shape,1))
    cnnarr = np.reshape(cnnarr, np.append(cnnarr.shape,1))

    return [
               bbarr[trainidx], cnnarr[trainidx], Plstmarr[trainidx],
               EClstmarr[trainidx], Slstmarr[trainidx],
               categarr[trainidx], montharr[trainidx], grouparr[trainidx], mktarr[trainidx], sipoarr[trainidx]
            ], \
           resultarr[trainidx], labelarr[trainidx], \
           [
               bbarr[testidx], cnnarr[testidx], Plstmarr[testidx],
               EClstmarr[testidx], Slstmarr[testidx],
               categarr[testidx], montharr[testidx], grouparr[testidx], mktarr[testidx], sipoarr[testidx]
            ], \
           resultarr[testidx], labelarr[testidx]

def build_multimodal(sourcearr):

    lstm_filter = 32

    input_bb = tf.keras.Input(shape=sourcearr[0].shape[1:], name='bb_input')
    x = keras.layers.BatchNormalization(axis=1)(input_bb)
    x = keras.layers.Dense(lstm_filter)(x)
    output_bb = keras.layers.Reshape((1, lstm_filter))(x)

    input_cnn = tf.keras.Input(shape=sourcearr[1].shape[1:], name='cnn_input')
    x = keras.layers.BatchNormalization(axis=1)(input_cnn)
    x = keras.layers.Conv2D(32, (3, 2), activation='relu')(x)
    x = keras.layers.Conv2D(32, (3, 2), activation='relu')(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(lstm_filter)(x)
    output_cnn = keras.layers.Reshape((1, lstm_filter))(x)

    input_Plstm = tf.keras.Input(shape=sourcearr[2].shape[1:], name='Plstm_input')
    x = keras.layers.BatchNormalization()(input_Plstm)
    x = keras.layers.Dense(lstm_filter)(x)
    output_Plstm = keras.layers.Reshape((1, lstm_filter))(x)

    input_EClstm = tf.keras.Input(shape=sourcearr[3].shape[1:], name='EClstm_input')
    x = keras.layers.BatchNormalization()(input_EClstm)
    x = keras.layers.GRU(lstm_filter, activation='relu', return_sequences=True)(x)
    x = keras.layers.GRU(lstm_filter, activation='relu', return_sequences=False)(x)
    output_EClstm = keras.layers.Reshape((1, lstm_filter))(x)

    input_Slstm = tf.keras.Input(shape=sourcearr[4].shape[1:], name='Slstm_input')
    x = keras.layers.BatchNormalization(axis=1)(input_Slstm)
    x = keras.layers.Conv2D(32, (3, 2), activation='relu')(x)
    x = keras.layers.Conv2D(32, (3, 2), activation='relu')(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(lstm_filter)(x)
    output_Slstm = keras.layers.Reshape((1, lstm_filter))(x)

    input_categ = tf.keras.Input(shape=sourcearr[-5].shape[1:], name='categ_input')
    input_month = tf.keras.Input(shape=sourcearr[-4].shape[1:], name='month_input')
    input_group = tf.keras.Input(shape=sourcearr[-3].shape[1:], name='group_input')
    input_mkt = tf.keras.Input(shape=sourcearr[-2].shape[1:], name='mkt_input')
    input_sipo = tf.keras.Input(shape=sourcearr[-1].shape[1:], name='sipo_input')

    input_total = keras.layers.Concatenate(name='merged_input', axis=1)([output_bb
                                                                       , output_cnn
                                                                       , output_Plstm
                                                                       , output_EClstm
                                                                       , output_Slstm
                                                                      ])

    input_sub = input_total
    input_res = CategoricalAttention_v2(name='mktAttention')([input_sub, input_mkt])
    input_total = input_res
    input_res = CategoricalAttention_v2(name='sipoAttention')([input_sub, input_sipo])
    input_total += input_res
    input_res = CategoricalAttention_v2(name='groupAttention')([input_sub, input_group])
    input_total += input_res
    input_res = CategoricalAttention_v2(name='sectorAttention')([input_sub, input_categ])
    input_total += input_res
    input_res = CategoricalAttention_v2(name='monthAttention')([input_sub, input_month])
    input_total += input_res
    input_total = keras.layers.Flatten()(input_total)

    x = keras.layers.Dense(128, activation='relu')(input_total)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dense(32, activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)

    for _ in range(11):
        x = resDense(32, 0.5)(x)

    output_total = keras.layers.Dense(2, activation='softmax', use_bias=False)(x)

    model_total = keras.models.Model(inputs=[input_bb, input_cnn, input_Plstm, input_EClstm, input_Slstm
                                            , input_categ, input_month, input_group, input_mkt, input_sipo
                                             ],
                               outputs=output_total,
                               name='total_Model'
                               )

    model_total.summary()

    return model_total

train_x, train_y, _, test_x, test_y, test_label= make_sourcearr(splitdate='2022-03-31', endsplit='2022-12-31')

model = build_multimodal(train_x)
#model = keras.models.load_model('./model_weight/bestweight/multimodal_class_best220214_1536.h5', custom_objects={'CategoricalAttention':CategoricalAttention,'resDense':resDense, 'f1_m':f1_m})
lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
    initial_learning_rate=0.0001,
    first_decay_steps=150,
    t_mul=1,
    m_mul=1
)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
                    loss='categorical_crossentropy',
                    metrics=['accuracy', f1_m]
              )

#save model architecture to image
#plot_model(model, to_file='class_model_plot.png', show_shapes=True, show_layer_names=True)
callback = CustomStopper(monitor='val_accuracy', patience=300, start_epoch=0, restore_best_weights=True)

model.fit(x=train_x, y=train_y, shuffle=True
          ,epochs=1000000, verbose=1, callbacks=[callback]
          ,validation_split= 0.3
          )

model.save('./model_weight/multimodal_class.h5')

model = keras.models.load_model('./model_weight/multimodal_class.h5', custom_objects={'CategoricalAttention_v2':CategoricalAttention_v2, 'resDense':resDense, 'f1_m':f1_m})

prediction = model.predict(test_x).round()

pd.DataFrame([test_label, np.argmax(prediction,axis=1), np.argmax(test_y,axis=1)]).T.to_csv('prediction.csv')
print('accuracy of model :: ', model.evaluate(test_x,test_y), ' f1_score :: ', f1_score(test_y, prediction, average='macro'))
print(confusion_matrix(np.argmax(test_y,axis=1),np.argmax(prediction,axis=1)))
