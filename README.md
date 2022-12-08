# [AI/빅데이터] AI를 활용한 공모주 투자전략
  
**2021년 국내 주식시장에서는 역대 최대 규모의 IPO가 이루어졌습니다. IPO 시장 과열에 대한 우려가 확대되고 있어 보다 효과적인 IPO 투자가 요구되는 시기입니다. 이에 AI 방법론을 활용하여 IPO 종목의 시가수익률과 상장 후 주가 추세에 대한 예측을 시도하였습니다.**

## I. 국내 IPO 시장 과열에 따른 공모주 주가 변동성 확대 우려

- 2021년 IPO 규모가 최대치를 달성하며 IPO 시장이 활황세를 보이고 있다. 공모주 투자는 안전하면서도 높은 수익을 기대할 수 있는 투자로 인식되고 있다.

- 다만, 시장의 과열 양상으로 공모주의 주가 변동성도 확대되고 있다. 이에 보다 효과적인 공모주 투자 전략을 수립할 필요가 있다.


## II. AI를 활용한 공모주 시가수익률 예측 모델과 이를 활용한 투자전략

- AI 기술을 활용해 공모주의 공모가 대비 시가수익률이 20% 이상일 확률을 측정하는 모델을 구현하였다. 검증 결과 모델의 정확도는 약 70%를 기록하였다.

- 모델이 상승 전망한 공모주에 투자하여 상장 첫날 시초가에 매도 시, 모든 IPO에 참여하는 경우 대비 투자수익률이 11.5%p 개선되었다. 또한 모델을 활용해 공모주의 시가상승률의 원인을 과잉수요와 저평가발행 요인으로 분해할 경우, 공모주의 1년 장기주가 전망에도 유효하였다.

# 
# <a border="0" href="http://tracking.nhqv.com/tracking?SITE_ID=4&amp;SEND_ID=3037338&amp;SCHD_ID=2206703&amp;WORKDAY=20220314&amp;TRACKING_CLOSE=2022-03-07&amp;TYPE=C&amp;CLICK_ID=003&amp;MEMBER_ID=a3lvdWppbi5raW1Abmhxdi5jb20=&amp;MEMBER_ID_SEQ=32612&amp;URL=https://download.nhqv.com/www/plugin/pdfjs/web/viewer.html?r=CommFile&amp;p=/cis/rsh/inv&amp;i=CISPPR20220314150959516" target="_blank" title="NH 리서치 원문보기"><img border="0" src="https://www.nhqv.com/img/ems/research/img_09.jpg"></a>

#
## Patch Note 2022-12-08

1. Attention Network 개선
- 기존 모델의 Categorical Feature Weighting Network는 학습된 Feature값들 전부에 대해 Attention 비중값을 계산
- 이로인해 학습해야하는 비중값의 개수가 지나치게 많아 학습 및 수렴에 어려움을 겪음
- 또한, 변수별 구분없이 모든 Feature값들을 취합하여 학습하므로 범주변수에 따른 변수별 민감도에 대한 측정의 신뢰도가 떨어짐

- 새로 고안한 Categorical Attention Network는 Feature별이 아닌 변수 구분별로 비중값을 학습하므로 학습해야 하는 비중값의 개수가 현저히 감소
- 변수 구분별로 학습된 비중값이므로 범주변수별 변수의 민감도 비교가 보다 신뢰도 상승
- 학습시 모델의 정확도의 수렴속도도 보다 빠르게 측정됨

- 민감도 외에도 범주변수 자체가 상승확률에 편향을 발생시킬 수 있음
- 따라서 범주변수별 편향도(Categorical Bias)를 직전 Network에서 더하여 범주변수에 따른 상승확률 편향도 반영 및 계산

2. 재무제표 누락값 처리
- FY-1, FY-2의 경우, 값이 누락 경우가 잦음. 누락값은 0으로 처리하여 Masking 처리
- 재주제표 Outlier값을 제외시키는 것은 되려 모델 예측력을 감소시킴. Outlier 처리는 추가하지 않음

3. LSTM Network 제외 및 CNN Network 개선
- LSTM Network 자체가 모델 학습 시간을 증가시키는 경향이 있을뿐더러, 입력변수들의 시계열 길이가 길지 않으므로 CNN Network를 통한 학습으로도 충분히 Feature값 출력 가능
- 동일 변수 내에서의 시계열 값만을 추상화시키기 위하여 1D Convolution Network를 적용하였으며 MaxPool 단계에서도 시계열 축에 대해서는 Pooling
