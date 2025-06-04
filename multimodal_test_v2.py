from customclass.custommod import resDense, CategoricalAttention_v2, f1_m
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from tensorflow import keras
from sklearn.metrics import confusion_matrix, f1_score
import os
from joblib import load
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def load_enc_transform(encname, arr):
    enc = load('./onehot_encoders/{}.joblib'.format(encname))
    onehotvector = enc.transform(np.array(arr).reshape(-1, 1)).toarray()

    return onehotvector

def make_sourcearr(file_path='./backdata/'):
    #import source data
    #encoder should be CP949 to display Korean
    bd_cnn = pd.read_csv(file_path+'cnnadj.csv', encoding='utf-8-sig', index_col='Code').T
    bd_cnn = bd_cnn.drop(labels=['Name','MKT Cap'], axis=1).astype('float')
    #clip outliers
    #upperbound = pd.read_csv(file_path+'upperbound.csv', encoding='CP949', index_col=0)
    #bd_cnn = bd_cnn.clip(lower=0, upper=upperbound.values.flatten(), axis=1)
    #bd_cnn = bd_cnn.clip(upper=bd_cnn.quantile(0.9), axis=1)

    bd_EClstm = pd.read_excel(file_path+'0.raw_Econ_monthly.xlsx',
                              sheet_name='final_data',
                              index_col=0, parse_dates=True,
                              ).fillna(value=0)

    bd_Slstm = pd.read_csv(file_path+'SWNEMO_Score.csv',
                           index_col=0, parse_dates=True,
                           encoding='utf-8-sig')
    bd_Slstm = bd_Slstm.rolling(13).mean().dropna()

    bd_Rlstm = pd.read_csv(file_path+'mktrtn.csv',
                           index_col=0, parse_dates=True,
                           encoding='utf-8-sig').pct_change().fillna(value=0)

    bd_bb = pd.read_csv(file_path+'BB.csv', encoding='utf-8-sig', index_col='종목코드')
    bd_bb.기업집단 = bd_bb.기업집단.apply(lambda x: 0 if x is np.nan else 1)

    match_comp = list(set(bd_bb.index) & set(bd_cnn.index))
    bd_cnn = bd_cnn.loc[match_comp]
    bd_bb = bd_bb.loc[match_comp].sort_values(by=['상장일', '종목명'], ascending=False)

    onehotE = OneHotEncoder()
    #make backdata
    bb_name = ['시총비중','신주비율','구주비율','유통비율','우리사주','기관투자자','일반투자자','상장주선수']
    labelarr = []
    bbarr = []
    cnnarr = []
    Plstmarr = []
    EClstmarr = []
    Slstmarr = []
    Rlstmarr = []
    categarr = []
    montharr = []
    grouparr = []
    mktarr = []
    sipoarr = []
    resultarr = []
    datearr = [] #for reindexing
    for _, eachrow in bd_bb.iterrows():
        #if label is not empty, skip number
        if eachrow['시가상승률']!=0:
            continue
        #define elements of company_idx
        compcode = eachrow.name
        groupnm = eachrow.기업집단
        bbdate = pd.to_datetime(eachrow['수요예측일'])

        #define datas
        temp_bb = eachrow[bb_name].values
        temp_plstm = bd_bb['시가상승률'][bd_bb['상장일']<eachrow['수요예측일']].iloc[:20].values
        temp_plstm = np.pad(temp_plstm, pad_width=(20-len(temp_plstm),0))
        temp_cnn = np.vstack([bd_cnn.loc[compcode].filter(regex='FY0').values,
                              bd_cnn.loc[compcode].filter(regex='FY-1').values,
                              bd_cnn.loc[compcode].filter(regex='FY-2').values]).T
        temp_EClstm = bd_EClstm.iloc[max(bd_EClstm.index.get_loc(bbdate, method='ffill') - 60,0):bd_EClstm.index.get_loc(bbdate,method='ffill')].T.values
        temp_Slstm = bd_Slstm.iloc[bd_Slstm.index.get_loc(bbdate, method='ffill') - 13:bd_Slstm.index.get_loc(bbdate,method='ffill')].T.values
        temp_Rlstm = bd_Rlstm.iloc[bd_Rlstm.index.get_loc(bbdate, method='ffill') - 13:bd_Rlstm.index.get_loc(bbdate,method='ffill')].T.values

        mktarr.append(eachrow['시장구분'])
        labelarr.append(compcode)
        bbarr.append(temp_bb)
        cnnarr.append(temp_cnn)
        Plstmarr.append(temp_plstm)
        EClstmarr.append(temp_EClstm)
        Slstmarr.append(temp_Slstm)
        Rlstmarr.append(temp_Rlstm)
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
    Rlstmarr = np.array(Rlstmarr).astype('float')
    sipoarr = np.array(sipoarr).astype('float')
    mktarr = load_enc_transform('mktenc', mktarr)
    categarr = load_enc_transform('categenc', categarr)
    montharr = load_enc_transform('monthenc', montharr)
    grouparr = load_enc_transform('groupenc', grouparr)
    sipoarr = load_enc_transform('sipoenc', sipoarr)

    resultarr = np.array(resultarr).astype('float')

    datearr_df = pd.DataFrame(range(len(datearr)), index=datearr)
    dateidx = datearr_df.sort_index(ascending=False).values.flatten()[:]
    EClstmarr = np.reshape(EClstmarr, np.append(EClstmarr.shape, 1))
    Rlstmarr = np.reshape(Rlstmarr, np.append(Rlstmarr.shape, 1))
    Slstmarr = np.reshape(Slstmarr, np.append(Slstmarr.shape,1))
    cnnarr = np.reshape(cnnarr, np.append(cnnarr.shape,1))

    return [
               bbarr[dateidx], cnnarr[dateidx], Plstmarr[dateidx],
               EClstmarr[dateidx], Slstmarr[dateidx], Rlstmarr[dateidx],
               categarr[dateidx], montharr[dateidx], grouparr[dateidx], mktarr[dateidx], sipoarr[dateidx]
            ], \
           resultarr[dateidx], labelarr[dateidx]

sourcearr, resultarr, label = make_sourcearr()

model = keras.models.load_model('./model_weight/multimodal_class_v2_2_best.h5',
                                custom_objects={'CategoricalAttention_v2':CategoricalAttention_v2, 'resDense':resDense, 'f1_m':f1_m}
                                )

#prediction = model.predict(sourcearr).round()
prediction = model.predict(sourcearr)
prediction_prob = pd.DataFrame(prediction)
predarr = np.array([np.random.choice([0,1],p=x) for x in prediction])

#prediction = OneHotEncoder().fit_transform(predarr.reshape(-1,1)).toarray()
prediction = model.predict(sourcearr).round()
prediction_label = pd.DataFrame([label, np.argmax(prediction,axis=1), np.argmax(resultarr,axis=1)]).T

pd.concat([prediction_label, prediction_prob], axis=1).to_csv('prediction.csv')

print('accuracy of model :: ', model.evaluate(sourcearr,resultarr), ' f1_score :: ', f1_score(resultarr, prediction, average='macro'))
print(confusion_matrix(np.argmax(resultarr,axis=1),np.argmax(prediction,axis=1)))