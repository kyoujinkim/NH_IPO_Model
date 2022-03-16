import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow.keras.utils import plot_model
from sklearn.preprocessing import OneHotEncoder
from customclass.custommod import resDense, CategoricalAttention, f1_m, CustomStopper
from matplotlib import font_manager, rc, use
from sklearn.metrics import confusion_matrix, f1_score
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

use("TkAgg")
#Korean font setting
font_path = 'C:/Windows/Fonts/gulim.ttc'
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)

def make_sourcearr(split, endsplit=0.1, consider_time=True, file_path='./backdata/'):
    #import source data
    #encoder should be CP949 to display Korean
    bd_cnn = pd.read_csv(file_path+'cnnadj.csv', encoding='CP949')
    bd_cnn.index = bd_cnn.Code
    bd_cnn = bd_cnn.drop('Code', axis=1)

    bd_lstm = pd.read_csv(file_path+'lstm.csv', encoding='CP949')
    bd_lstm.index = pd.to_datetime(bd_lstm.DATE)
    bd_lstm = bd_lstm.drop('DATE', axis=1).dropna()

    bd_EClstm = pd.read_csv(file_path+'econ.csv', encoding='CP949')
    bd_EClstm.index = pd.to_datetime(bd_EClstm.DATE)
    bd_EClstm = bd_EClstm.drop('DATE', axis=1).fillna(value=0)

    bd_Slstm = pd.read_csv(file_path+'SentScore.csv', encoding='CP949')
    bd_Slstm.index = pd.to_datetime(bd_Slstm.DATE)
    bd_Slstm = bd_Slstm.drop('DATE', axis=1).dropna()
    bd_Slstm = bd_Slstm.rolling(13).mean().dropna()

    bd_IVlstm = pd.read_csv(file_path+'IVlstm.csv', encoding='CP949')
    bd_IVlstm.index = pd.to_datetime(bd_IVlstm.DATE)
    bd_IVlstm = bd_IVlstm.drop('DATE', axis=1).dropna()

    bd_bb = pd.read_csv(file_path+'BB.csv', encoding='CP949')
    bd_bb.기업집단 = bd_bb.기업집단.apply(lambda x: 0 if x is np.nan else 1)

    onehotE = OneHotEncoder()
    #make backdata
    bb_name = ['시총비중','신주비율','구주비율','유통비율','우리사주','기관투자자','일반투자자','상장주선수'
               ]
    labelarr = []
    bbarr = []
    cnnarr = []
    lstmarr = []
    Plstmarr = []
    EClstmarr = []
    Slstmarr = []
    categarr = []
    montharr = []
    IVlstmarr = []
    grouparr = []
    mktarr = []
    sipoarr = []
    resultarr = []
    for idx in range(len(bd_bb)):
        #if label is empty, skip number
        if pd.isna(bd_bb['시가상승률'].iloc[idx]):
            continue
        #define elements of company_idx
        compcode = bd_bb['종목코드'].iloc[idx]
        groupnm = bd_bb.기업집단.iloc[idx]
        bbdate = pd.to_datetime(bd_bb['수요예측일'].iloc[idx])

        #define datas
        temp_bb = bd_bb[bb_name].iloc[idx].values

        temp_plstm = bd_bb['시가상승률'][pd.to_datetime(bd_bb['상장일'])<pd.to_datetime(bd_bb['수요예측일'].iloc[idx])].iloc[:10]
        temp_plstm = np.append([0]*(10-len(temp_plstm)),temp_plstm)

        temp_cnn = np.vstack([bd_cnn[compcode][2:13].values,bd_cnn[compcode][13:24].values,bd_cnn[compcode][24:35].values]).T

        temp_lstm = bd_lstm[['시총대비예탁']].iloc[bd_lstm.index.get_loc(bbdate, method='ffill')-20:bd_lstm.index.get_loc(bbdate, method='ffill')].values

        temp_EClstm = bd_EClstm.iloc[max(bd_EClstm.index.get_loc(bbdate, method='ffill') - 12,0):bd_EClstm.index.get_loc(bbdate,method='ffill')].values

        temp_Slstm = bd_Slstm.iloc[bd_Slstm.index.get_loc(bbdate, method='ffill') - 13:bd_Slstm.index.get_loc(bbdate,method='ffill')].values

        temp_IVlstm = bd_IVlstm.iloc[bd_IVlstm.index.get_loc(bbdate, method='ffill') - 20:bd_IVlstm.index.get_loc(bbdate,method='ffill')].values

        mktarr.append(bd_bb['시장구분'].iloc[idx])
        labelarr.append(bd_bb['종목코드'].iloc[idx])
        bbarr.append(temp_bb)
        cnnarr.append(temp_cnn)
        lstmarr.append(temp_lstm)
        Plstmarr.append(temp_plstm)
        EClstmarr.append(temp_EClstm)
        Slstmarr.append(temp_Slstm)
        IVlstmarr.append(temp_IVlstm)
        categarr.append(bd_bb['W_Sector'].iloc[idx])
        montharr.append(pd.to_datetime(bd_bb['수요예측일']).iloc[idx].month)
        grouparr.append(groupnm)
        sipoarr.append(bd_bb['특례상장'].iloc[idx])
        resultarr.append(np.array([0,1])) if bd_bb['시가상승률'].iloc[idx]>1.2 else resultarr.append(np.array([1,0]))

    labelarr = np.array(labelarr)
    bbarr = np.array(bbarr).astype('float')
    cnnarr = np.array(cnnarr).astype('float')
    lstmarr = np.array(lstmarr).astype('float')
    Plstmarr = np.array(Plstmarr).astype('float')
    EClstmarr = np.array(EClstmarr).astype('float')
    Slstmarr = np.array(Slstmarr).astype('float')
    IVlstmarr = np.array(IVlstmarr).astype('float')
    sipoarr = np.array(sipoarr).astype('float')
    mktarr = onehotE.fit_transform(np.array(mktarr).reshape(-1, 1)).toarray()
    categarr = onehotE.fit_transform(np.array(categarr).reshape(-1,1)).toarray()
    montharr = onehotE.fit_transform(np.array(montharr).reshape(-1,1)).toarray()
    grouparr = onehotE.fit_transform(np.array(grouparr).reshape(-1,1)).toarray()
    sipoarr = onehotE.fit_transform(np.array(sipoarr).reshape(-1,1)).toarray()

    resultarr = np.array(resultarr).astype('float')

    Plstmarr = np.reshape(Plstmarr, np.append(Plstmarr.shape,1))
    cnnarr = np.reshape(cnnarr, np.append(cnnarr.shape,1))

    if consider_time == False:
        shf_idx = np.arange(len(bbarr))
        np.random.seed(111)
        np.random.shuffle(shf_idx)
        labelarr = labelarr[shf_idx]
        bbarr = bbarr[shf_idx]
        cnnarr = cnnarr[shf_idx]
        lstmarr = lstmarr[shf_idx]
        Plstmarr = Plstmarr[shf_idx]
        EClstmarr = EClstmarr[shf_idx]
        Slstmarr = Slstmarr[shf_idx]
        IVlstmarr = IVlstmarr[shf_idx]
        categarr = categarr[shf_idx]
        montharr = montharr[shf_idx]
        grouparr = grouparr[shf_idx]
        mktarr = mktarr[shf_idx]
        sipoarr = sipoarr[shf_idx]
        resultarr = resultarr[shf_idx]

    splitnum = round(len(bbarr)*split)
    Esplitnum = round(len(bbarr)*(split-endsplit))

    return [bbarr[splitnum:], cnnarr[splitnum:], lstmarr[splitnum:], Plstmarr[splitnum:], EClstmarr[splitnum:], Slstmarr[splitnum:],IVlstmarr[splitnum:]
            , categarr[splitnum:], montharr[splitnum:], grouparr[splitnum:], mktarr[splitnum:], sipoarr[splitnum:]], \
           resultarr[splitnum:], labelarr[splitnum:], \
           [bbarr[Esplitnum:splitnum], cnnarr[Esplitnum:splitnum], lstmarr[Esplitnum:splitnum], Plstmarr[Esplitnum:splitnum], EClstmarr[Esplitnum:splitnum], Slstmarr[Esplitnum:splitnum], IVlstmarr[Esplitnum:splitnum]
            , categarr[Esplitnum:splitnum], montharr[Esplitnum:splitnum], grouparr[Esplitnum:splitnum], mktarr[Esplitnum:splitnum], sipoarr[Esplitnum:splitnum]], \
           resultarr[Esplitnum:splitnum], labelarr[Esplitnum:splitnum]

def build_multimodal(sourcearr):

    lstm_filter = 32

    input_bb = tf.keras.Input(shape=sourcearr[0].shape[1:], name='bb_input')
    input_bb = layers.BatchNormalization(axis=1)(input_bb)

    input_cnn = tf.keras.Input(shape=sourcearr[1].shape[1:], name='cnn_input')
    x = layers.BatchNormalization(axis=1)(input_cnn)
    x = layers.Conv2D(32, (3, 1), activation='relu')(x)
    x = layers.Conv2D(32, (3, 1), activation='relu')(x)
    x = layers.Conv2D(32, (3, 2), activation='relu')(x)
    x = layers.Conv2D(32, (3, 2), activation='relu')(x)
    output_cnn = layers.Flatten()(x)

    input_lstm = tf.keras.Input(shape=sourcearr[2].shape[1:], name='lstm_input')
    x = layers.BatchNormalization()(input_lstm)
    x = layers.GRU(lstm_filter, activation='relu', return_sequences=True)(x)
    x = layers.GRU(lstm_filter, activation='relu', return_sequences=False)(x)
    output_lstm = layers.Flatten()(x)

    input_Plstm = tf.keras.Input(shape=sourcearr[3].shape[1:], name='Plstm_input')
    x = layers.BatchNormalization()(input_Plstm)
    x = layers.GRU(lstm_filter, activation='relu', return_sequences=True)(x)
    x = layers.GRU(lstm_filter, activation='relu', return_sequences=False)(x)
    output_Plstm = layers.Flatten()(x)

    input_EClstm = tf.keras.Input(shape=sourcearr[4].shape[1:], name='EClstm_input')
    x = layers.BatchNormalization()(input_EClstm)
    x = layers.GRU(lstm_filter, activation='relu', return_sequences=True)(x)
    x = layers.GRU(lstm_filter, activation='relu', return_sequences=False)(x)
    output_EClstm = layers.Flatten()(x)

    input_Slstm = tf.keras.Input(shape=sourcearr[5].shape[1:], name='Slstm_input')
    x = layers.BatchNormalization()(input_Slstm)
    x = layers.GRU(lstm_filter, activation='relu', return_sequences=True)(x)
    x = layers.GRU(lstm_filter, activation='relu', return_sequences=False)(x)
    output_Slstm = layers.Flatten()(x)

    input_IVlstm = tf.keras.Input(shape=sourcearr[6].shape[1:], name='IVlstm_input')
    x = layers.BatchNormalization()(input_IVlstm)
    x = layers.GRU(lstm_filter, activation='relu', return_sequences=True)(x)
    x = layers.GRU(lstm_filter, activation='relu', return_sequences=False)(x)
    output_IVlstm = layers.Flatten()(x)

    input_categ = tf.keras.Input(shape=sourcearr[7].shape[1:], name='categ_input')
    input_month = tf.keras.Input(shape=sourcearr[8].shape[1:], name='month_input')
    input_group = tf.keras.Input(shape=sourcearr[9].shape[1:], name='group_input')
    input_mkt = tf.keras.Input(shape=sourcearr[10].shape[1:], name='mkt_input')
    input_sipo = tf.keras.Input(shape=sourcearr[11].shape[1:], name='sipo_input')

    input_total = layers.Concatenate(name='merged_input')([input_bb,
                                                           output_cnn,
                                                           output_lstm,
                                                           output_Plstm,
                                                           output_EClstm,
                                                           output_Slstm,
                                                           output_IVlstm
                                                          ])

    input_sub = input_total
    input_res = CategoricalAttention(name='mktAttention')([input_sub, input_mkt])
    input_total = input_res
    input_res = CategoricalAttention(name='sipoAttention')([input_sub, input_sipo])
    input_total += input_res
    input_res = CategoricalAttention(name='groupAttention')([input_sub, input_group])
    input_total += input_res
    input_res = CategoricalAttention(name='sectorAttention')([input_sub, input_categ])
    input_total += input_res
    input_res = CategoricalAttention(name='monthAttention')([input_sub, input_month])
    input_total += input_res

    x = layers.Dense(512, activation='relu')(input_total)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(32, activation='relu')(x)
    x = layers.BatchNormalization()(x)

    for _ in range(11):
        x = resDense(32, 0.5)(x)

    output_total = layers.Dense(2, activation='softmax', use_bias=False)(x)

    model_total = models.Model(inputs=[input_bb, input_cnn, input_lstm, input_Plstm, input_EClstm, input_Slstm, input_IVlstm,
                                       input_categ, input_month, input_group, input_mkt, input_sipo],
                               outputs=output_total,
                               name='total_Model'
                               )

    model_total.summary()

    return model_total

sourcearr, resultarr, _, test_sourcearr, test_resultarr, test_label= make_sourcearr(split=0.3, endsplit=0.3, consider_time=True)

model = build_multimodal(sourcearr)
lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
    initial_learning_rate=0.0003,
    #alpha=0.001,
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

model.fit(x=sourcearr, y=resultarr, shuffle=True
          ,epochs=1000000, verbose=1, callbacks=[callback]
          ,validation_split= 0.3
          )

model.save('./model_weight/multimodal_class.h5')

#model = models.load_model('./model_weight/multimodal_class.h5', custom_objects={'CategoricalAttention':CategoricalAttention, 'resDense':resDense, 'f1_m':f1_m})

prediction = model.predict(test_sourcearr).round()
print('accuracy of model :: ', model.evaluate(test_sourcearr,test_resultarr), ' f1_score :: ', f1_score(test_resultarr, prediction, average='macro'))
print(confusion_matrix(np.argmax(test_resultarr,axis=1),np.argmax(prediction,axis=1)))