<div align=center>

![musicreview](https://github.com/5solemi5/sentiment_analysis/assets/104000117/78394c9e-85e3-4048-b273-37448e01be1e)

  
# 📀음악 리뷰 감성분석📀 
  
**MobileBert를 활용한 긍부정 예측 딥러닝 프로젝트**
  
음악 리뷰에는 보통 긍정적인 리뷰가 많지만, 일부 부정적인 리뷰도 있다. 이를 이진 분류 문제로 정의하여 MobileBERT 모델을 훈련시킨다. 
  
<h2>:heavy_check_mark:Tech Stack</h2>
<img src="https://img.shields.io/badge/PyTorch-E34F26?style=flat-square&logo=PyTorch&logoColor=white"/></a> 
<img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=Python&logoColor=white"/></a>

</div>

# 1. 서론
## 1.1 음악 플랫폼 시장

 ![c](https://github.com/5solemi5/sentiment_analysis/assets/104000117/d5310985-abcf-4d17-b0ee-a27c74fe5985)
 ![KakaoTalk_20230621_105756684](https://github.com/5solemi5/sentiment_analysis/assets/104000117/750655f9-18b1-4bda-b451-80b7d04d473b)
 
 <div align=center>
  
[[자료: RYM사이트]](https://rateyourmusic.com/)
  
 </div>



  현재 수많은 사람들이 음악 스트리밍 플랫폼을 통해 인터넷에 연결된 장치에서 음악을 듣고, 저장 및 다운로드 없이 해당 음악에 대한 액세스 권한을 얻을 수 있다. 때문에 다양한 음악에 대한 접근이 쉬워졌으며, 이러한 플랫폼들은 인터넷의 보급과 함께 급속도로 성장하고 있다. 전 세계 음악 스트리밍 매출액은 '12년 7.3억 달러에서 '17년 66억 달러로 연평균 55.2% 증가하였다. [<sup>[1]</sup>](https://test.hri.co.kr/upload/board/201921514759[1].hwp) 
  
  기존의 음악 스트리밍 플랫폼은 단순히 음악을 스트리밍하기 위한 플랫폼이었지만, 지금은 다양한 기능들을 제공하며 기본적으로 댓글과 평점과 같이 청취자가 실제로 앨범에 대해 피드백을 줄 수 있는 환경이 있다. 그리고 스트리밍 플랫폼 외에도 AllMusic, Pitchfork, RYM 등과 같은 음악 관련 웹사이트나 앱에서도 앨범에 대한 리뷰와 평점을 남길 수 있고 이러한 서비스들이 많아지고 있다. 음악 플랫폼의 시장이 확장되고 음악 리뷰 데이터를 얻을 수 있는 환경이 많아지는 현재, 다양한 음악 플랫폼에 많은 음악 리뷰 데이터들이 쌓이고 있다. 

  대표적인 음악 스트리밍 플랫폼으로 Spotify가 있는데, Spotify에만 2023년 3월 31일 기준으로 5억 1,500만 명이 오디오 스트리밍 서비스를 이용하고 있고 Spotify는 거의 모든 연령대 뿐만 아니라 선진국과 개발도상국 시장 모두에서 이용자 수에 대한 큰 성장을 보였다. 이러한 성장의 대부분은 무료 광고 지원 버전의 Spotify 서비스를 사용하는 사람들을 기반으로 한다. 프리미엄 구독은 전 분기 대비 2%, 전년 대비 15% 증가하여 2억 5백만에서 2억 1천만으로 전체 성장 속도를 따라가지 못했다. 그럼에도 불구하고 프리미엄 가입자는 Spotify가 투자자 지침에서 지적한 것보다 300만 명 더 증가했다. [<sup>[2]</sup>](https://www.engadget.com/spotify-reaches-more-than-half-a-billion-users-for-the-first-time-142818686.html)   


<div align=center><img src = "https://github.com/5solemi5/sentiment_analysis/assets/104000117/85ce4a6e-d94e-45db-a2eb-05d64dd82311" width="580">
 
  [자료: [국제음반산업협회(IFPI)](https://test.hri.co.kr/upload/board/201921514759[1].hwp), 2022-2023년 Spotify시장 분석 그래프]
  
</div>

 Net Operating Loss 선 그래프 (순 영업 손실):
전반적으로 Spotify는 해당 분기에 1억 5,600만 유로(1억 7,200만 달러)의 순 영업 손실을 기록했다. 이는 2022년 1분기에 본 600만 유로(660만 달러) 손실보다 훨씬 많은 금액이지만, 스포티파이가 지난 분기에 2억 7000만 유로(2억 9700만 달러) 손실을 낸 것보다 개선된 것이다. [<sup>[2]</sup>](https://www.engadget.com/spotify-reaches-more-than-half-a-billion-users-for-the-first-time-142818686.html)

 Revenue 선 그래프 (매출):
이 그래프는 분기별 매출을 나타내며, Q1 2023까지의 데이터를 포함한다. 매출은 2022년 Q1부터 2023년 Q1까지 꾸준히 증가하는 추세를 보인다. 하지만, Q4 2022와 Q1 2023 사이에 약간의 하락이 있었음을 알 수 있다.[<sup>[3]</sup>](https://newsroom.spotify.com/2023-04-25/spotify-reports-first-quarter-2023-earnings/)

 Ad-supported Revenue 막대 그래프 (광고 매출):
이 그래프는 분기별 광고 매출을 나타낸다. Q1 2022부터 Q1 2023까지 광고 매출은 상승했지만, Q4 2022와 Q1 2023 사이에 약간의 감소가 있었다. [<sup>[3]</sup>](https://newsroom.spotify.com/2023-04-25/spotify-reports-first-quarter-2023-earnings/)
   
 
## 1.2 문제정의

 <div align=center><img src = "https://github.com/5solemi5/sentiment_analysis/assets/104000117/3eee0ccc-1c2c-483d-9345-767b5d5b3c76" width="970">
  
[[자료: Cyanite사이트의 How Do AI Music Recommendation Systems Work 그림]](https://cyanite.ai/2021/09/02/how-do-ai-music-recommendation-systems-work/)
   
 </div>
 
  이미 Spotify, Apple Music, Vibe 등의 플랫폼에서 사용자가 들은 음악에 대한 정보, 재생 횟수, 스킵 여부 등의 데이터들을 이용한 AI 서비스가 배포되고 있는 상황이다.[<sup>[4]</sup>](https://cyanite.ai/2021/09/02/how-do-ai-music-recommendation-systems-work/) 이처럼 음악 리뷰 데이터도 AI 기술을 통해 유의미하게 이용될 수 있다. 
  
  과거에는 음악평론가들이 주로 음악 리뷰를 작성했지만, 최근에는 일반 사용자들도 자유롭게 리뷰를 작성할 수 있는 환경이 되었다. 이렇게 모인 음악 리뷰 데이터는 대중의 선호도와 음악 산업의 동향 파악에 유용한 정보를 제공한다. 이 데이터에 감성분석 모델을 적용하면, 이 데이터를 분석하여 음악 추천 시스템을 개발하거나, 음악 장르와 아티스트의 인기도 등을 파악하여 마케팅 전략을 수립할 수 있다.  
  
  이는 음악 산업에서 매우 중요한 분야가 될 수 있으며, 다양한 역량을 강화하는 데에도 도움이 된다. 때문에 본인은 첫걸음으로 음악 리뷰 데이터를 사용해서 MobileBert를 활용한 긍부정 예측 딥러닝 프로젝트를 해볼 예정이다.

# 2.데이터
## 2.1 원시 데이터

- 원시 데이터 출처:

  Kaggle에서 제공하는 [Music Album Reviews and Ratings Dataset](https://www.kaggle.com/datasets/michaelbryantds/78k-music-album-reviews) 데이터셋을 이용한다. 이 데이터 세트는 2022년 5월에 [RYM사이트](https://rateyourmusic.com/)에서 스크래핑했다. 

![d](https://github.com/5solemi5/sentiment_analysis/assets/104000117/cc1037d2-4e78-400f-96a0-bcc7556a22cd)

<div align=center>
  
[[자료: RYM사이트 리뷰의 일부분]](https://rateyourmusic.com/)
  
</div>
  
- 원시 데이터 분석:

<div align=center>

|Index|Review|Rating|
|-|-|-|
|1|i think i actually under-rate ok computer...|5|
|2|when i was 15 and it was maybe the fourth...|5|
|3|atmospheric a rock anthem as the band would...|4.5|
|...|...|...|
|80279|i do not like funk and i do not like prince. this is ...|1|

[자료: 원시 데이터의 형태]

</div>

80,279여개의 Review가 있고 Rating은 부정에서 긍정을 0~5사이에서 0.5 단위로 점수를 매겼다. 원시 데이터에서 NaN 값을 제거한 데이터의 수는80,245개이다. 앞으로 NaN 값을 제거한 데이터를 가지고 분석 할 것이다.
    
  <div align=center><img src = "https://github.com/5solemi5/sentiment_analysis/assets/104000117/677827f2-88b4-4d44-a5bb-08b03a516fda" width="600">
 
  [자료: 원시 데이터_Review의 개수와 Rating 관계분석 그래프]
  
  </div>

  위 그래프를 통해 Rating의 긍정 부분에 Review 수가 치우쳐 있고 평점이 높을수록 리뷰의 개수가 증가하고 있음을 알 수 있다.
가장 많은 리뷰 개수는 평점 5에 해당하는 앨범들이 차지하고 있다. 이는 평점 5를 받은 앨범들이 가장 인기가 많거나, 사용자들이 긍정적으로 평가한 앨범들이 많다는 것을 알 수 있고 평점이 2.5 미만인 앨범들에 대한 리뷰가 상대적으로 적기 때문에 이는 반대로 낮은 평점을 받은 앨범들에 대한 관심이 적거나, 리뷰가 적게 작성된 것을 의미할 수 있다.

  <div align=center>
    <img src = "https://github.com/5solemi5/sentiment_analysis/assets/104000117/e0316a3f-9ae4-4e01-9368-08a031ef29fa" width="800">
    <img src = "https://github.com/5solemi5/sentiment_analysis/assets/104000117/0fca3c3d-a515-499a-9313-189f4dd221c5" width="800">
 
  [자료: 원시 데이터_Review의 문장길이와 개수 관계분석 그래프]
  
  </div>

  리뷰의 대부분은 길이가 0에서 5,000 사이에 분포하고 있고 길이가 5,000을 넘어가면서부터 리뷰의 개수가 급격히 감소한다.

## 2.2 분석 데이터

딥러닝 학습을 위해 원본 데이터를 가공하여 분석 데이터를 만든다.

- 가공한 방식:
  
(1) 학습하는 Review의 너무 길거나 짧은 문장들은 딥러닝 학습에 무의미한 데이터들이므로 제거하는 과정을 거친다. 문장 길이가 20~855인 Reviews만 추출했다. 

(2)Rating 3.5의 리뷰는 중립적인 내용의 리뷰가 대부분이었다. 학습 정확도를 높이기 위해 Rating 3.5의 리뷰는 제외했다. 최종 전처리한 데이터 수 48,844 건이다.

(3) 전처리한 데이터의 Rating이 4이상인 리뷰를 1(긍정), 3이하인 리뷰를 0(부정)으로 바꾸어서 이진 분류했다.

- 가공한 결과:
  
<div align=center>

  |Index|Review|Rating|
|-|-|-|
|1|this album is great.|1|
|2|it is thee illmatic.|1|
|3|son of shivaaaaaaaaa|1|
|...|...|...|
|48845|the replacements third album is widely regarded...|0|
  
[자료: 분석 데이터의 형태]

</div>

<div align=center> <img src = "https://github.com/5solemi5/sentiment_analysis/assets/104000117/0678ea91-e667-44f9-bc2a-8b38b82a469c" width="600">
 
  [자료: 분석 데이터_Review의 개수와 Rating 관계분석 그래프]
   
  </div>

Rating이 4이상인 리뷰를 1(긍정), 3이하인 리뷰를 0(부정)으로 평균값보다 높은 임계값을 기준으로 이진 분류했음에도 불구하고 1(긍정)에 데이터가 치우쳐 있다.
클래스 불균형은 모델의 학습에 부정적인 영향을 미칠 수 있는 다음과 같은 이유로 인해 문제가 될 수 있다.

(1)	편향된 학습: 학습 데이터에서 많은 수의 샘플이 있는 클래스에 대해 모델이 더 많은 경험을 하게 된다. 이는 모델이 훈련 데이터의 분포에 따라 편향되게 학습될 수 있음을 의미한다. 편향된 학습은 모델이 소수 클래스의 패턴과 특징을 제대로 학습하지 못하게 하며, 새로운 샘플을 예측할 때 정확도와 성능을 저하시킬 수 있다.

(2)오분류 문제: 모델은 자연스럽게 많은 클래스에 속하는 샘플을 예측하는 경향이 있다. 이로 인해 소수 클래스의 샘플을 정확하게 예측하지 못하고 오분류할 가능성이 높아진다. 

(3)성능 측정의 오류: 클래스 불균형이 있는 데이터에서 모델의 성능을 평가하는 경우, 정확도(Accuracy)만으로는 모델의 성능을 정확히 평가할 수 없다. 클래스 불균형 데이터에서는 대다수 클래스로 예측하는 경향이 있기 때문에, 모델이 불균형한 데이터에서도 높은 정확도를 보일 수 있다.[<sup>[5]</sup>](https://towardsdatascience.com/handling-imbalanced-datasets-in-deep-learning-f48407a0e758)   

<div align=center> <img src = "https://github.com/5solemi5/sentiment_analysis/assets/104000117/2d860b8d-d3d1-4b25-986d-f2aeaaabe5b2" width="850">
 
  [자료: 분석 데이터_Review의 개수와 Rating 관계분석 그래프]

  대부분의 리뷰 길이가 0에서 300 사이에 집중되어 있다. 
   
  </div>

## 2.3 학습 데이터  

학습 데이터는 모델이 패턴과 특징을 학습하는 데 사용되는 데이터로, 이를 통해 모델은 입력과 출력 간의 관계를 학습하고 예측을 수행할 수 있게 된다. 학습 정확도를 높이기 위해 앞서 가공한 분석 데이터에서 발생하는 클래스 불균형을 해결한다.
  
- 추출한 방식: 1(긍정), 0(부정)에서 임의로 각각 1,000건씩 추출하여 2,000건을 학습했다.

- 추출한 결과:

<div align=center> 
  
  |Index|Review|Rating|
|-|-|-|
|1|hurrah finally i had my first radiohead experience...|0|
|2|so let me get this straight. a bunch of one-hit-wonde...|0|
|3|i can sort of understand the praise this album gets...|0|
|...|...|...|
|2000|8 out of the 12 songs are solid solid solid. the album...|1|
 
  [자료: 학습 데이터의 형태]
   
  </div>


# 3. 결과

## 3.1 MobileBERT를 사용한 결과
 Learning Curve (학습 곡선)은 보통 Train set과 Validation(test) set에 대해서 각각 loss와 metric을 훈련 중간중간 마다 체크한 곡선을 말한다. loss, metric을 체크하면 지금 모델이 underfit 되고 있는지, overfit 되고 있는지, 또는 그 외의 문제가 있는지를 알 수 있다. 
 
<div align=center> <img src = "https://github.com/5solemi5/sentiment_analysis/assets/104000117/6df6a480-c69d-476f-acca-ca204a61e1ea" width="750">
 
  [자료: training, validation 그래프]
  </div>

위의 그래프를 보면 loss는 계속 떨어지고, Accuracy는 높아지고 있다. 그렇다면 훈련이 잘 진행되고 있고, 언더피팅과 오버피팅이 일어나지 않았음을 알 수 있다.  
훈련 데이터 전체에 적용한 모델의 accuracy는 0.76%이다. 즉, 예측 모델이 입력 데이터를 올바르게 분류한 비율이 0.76%라는 의미이다.

## 3.2 분석 데이터 전체에 적용한 결과 
<div align=center> <img src = "https://github.com/5solemi5/sentiment_analysis/assets/104000117/936e66f7-d9c9-4101-b87a-15626699ea34" width="450">
 
  [자료: accuracy 결과창]
   
  </div>

분석 데이터 전체에 적용한 모델의 accuracy도 0.76%이다. 이 결과로부터 현재 사용 중인 예측 모델이 분석 데이터에 대해 그렇게 높지 않은 성능을 보인다는 것을 알 수 있다. 
  
## ⚒️개발환경⚒️

<img src="https://img.shields.io/badge/pycharm 2022.3.3-6495ED?style=flat-square&logo=pycharm&logoColor=white"/></a>

<img src="https://img.shields.io/badge/Python 3.9.0-6495ED?style=flat-square&logo=Python&logoColor=white"/></a>

<img src="https://img.shields.io/badge/pandas 1.4.4-6495ED?style=flat-square&logo=pandas&logoColor=white"/></a>

<img src="https://img.shields.io/badge/torch 1.12.1-6495ED?style=flat-square&logo=torch&logoColor=white"/></a>

<img src="https://img.shields.io/badge/tensorflow 2.9.1-6495ED?style=flat-square&logo=tensorflow&logoColor=white"/></a>

<img src="https://img.shields.io/badge/numpy 1.24.2-6495ED?style=flat-square&logo=numpy&logoColor=white"/></a>

<img src="https://img.shields.io/badge/transformers 4.21.2-6495ED?style=flat-square&logo=transformers&logoColor=white"/></a>

<img src="https://img.shields.io/badge/scikit-learn 1.2.2-6495ED?style=flat-square&logo=scikit-learn&logoColor=white"/></a>


# 4. 최종결론
딥러닝 학습 과정에서의 문제는 없었지만 accuracy가 낮은 것을 보아하니 입력 데이터의 정확성이 불완전했다고 판단된다. 음악 감상평은 종종 작품의 특성과 성질을 나타내기 위해 해석의 가능성이 많은 시적인 표현, 상징적인 비유, 예술 용어 등 예술적 언어와 비유를 사용한다. 또한 문장이 긴 리뷰들은 대체로 음악 앨범에 대한 평가보단 해석에 가까운 성격을 띄었다. 때문에 이번 프로젝트에 이용되었던 데이터 셋의 성격상 리뷰 글에 대한 점수의 연관성이 떨어질 수 밖에 없었다. 하지만 accuracy를 높이기 위한 다양한 데이터 전처리 시도들은 의미있는 경험이었다. 그리고 데이터 불균형이 모델 학습에 미치는 영향과 이유에 대해 배울 수 있었고 불균형을 처리하는 여러가지 방법들에 대해 알 수 있었던 시간이었다. 앞으로 딥러닝을 더 깊게 학습한 후, 나은 결과를 얻을 수 있도록 다음 프로젝트에 도전해보고자 한다.

# Reference

[1]https://test.hri.co.kr/upload/board/201921514759[1].hwp

[2]https://www.engadget.com/spotify-reaches-more-than-half-a-billion-users-for-the-first-time-142818686.html

[3]https://newsroom.spotify.com/2023-04-25/spotify-reports-first-quarter-2023-earnings/

[4]https://cyanite.ai/2021/09/02/how-do-ai-music-recommendation-systems-work/

[5]https://towardsdatascience.com/handling-imbalanced-datasets-in-deep-learning-f48407a0e758
