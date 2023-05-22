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
## 1.1 음악 플랫폼 시장과 음악 리뷰 데이터의 증가

  현재 수많은 사람들이 음악 스트리밍 플랫폼을 통해 인터넷에 연결된 장치에서 음악을 듣고, 저장 및 다운로드 없이 해당 음악에 대한 액세스 권한을 얻을 수 있다. 때문에 다양한 음악에 대한 접근이 쉬워졌으며, 이러한 플랫폼들은 인터넷의 보급과 함께 급속도로 성장하고 있다.
  
  <div align=center><img src = "https://github.com/5solemi5/sentiment_analysis/assets/104000117/1b33e7fc-d9ef-48c2-9ab9-d98c0dead007" width="300">
  
  [[자료: 국제음반산업협회(IFPI)]](https://test.hri.co.kr/upload/board/201921514759[1].hwp)
  
  </div>
  
  전 세계 음악 스트리밍 매출액은 '12년 7.3억 달러에서 '17년 66억 달러로 연평균 55.2% 증가하였다. [<sup>[1]</sup>](https://test.hri.co.kr/upload/board/201921514759[1].hwp) 
  
  <div align=center><img src = "https://github.com/5solemi5/sentiment_analysis/assets/104000117/e798581e-837d-40ab-b8eb-2e8968338dfe" width="800">
  
[[자료: Spotify Reports First Quarter 2023 Earnings 이미지]](https://newsroom.spotify.com/2023-04-25/spotify-reports-first-quarter-2023-earnings/)
  
  </div>
  
  그리고 대표적인 음악 스트리밍 플랫폼으로 Spotify가 있는데, Spotify에만 2023년 3월 31일 기준으로 5억 1,500만 명이 오디오 스트리밍 서비스를 이용하고 있고 Spotify는 거의 모든 연령대 뿐만 아니라 선진국과 개발도상국 시장 모두에서 이용자 수에 대한 큰 성장을 보였다. 이러한 성장의 대부분은 무료 광고 지원 버전의 Spotify 서비스를 사용하는 사람들을 기반으로 한다. 프리미엄 구독은 전 분기 대비 2%, 전년 대비 15% 증가하여 2억 5백만에서 2억 1천만으로 전체 성장 속도를 따라가지 못했다. 그럼에도 불구하고 프리미엄 가입자는 Spotify가 투자자 지침에서 지적한 것보다 300만 명 더 증가했다. [<sup>[2]</sup>](https://www.engadget.com/spotify-reaches-more-than-half-a-billion-users-for-the-first-time-142818686.html)   

<div align=center><img src = "https://github.com/5solemi5/sentiment_analysis/assets/104000117/d07094c2-8313-4229-9cd1-f7395f4607eb" width="850">

[자료: 2022-2023년 Spotify시장 분석 그래프]
  
</div>

  - Net Operating Loss 선 그래프 (순 영업 손실):
전반적으로 Spotify는 해당 분기에 1억 5,600만 유로(1억 7,200만 달러)의 순 영업 손실을 기록했다. 이는 2022년 1분기에 본 600만 유로(660만 달러) 손실보다 훨씬 많은 금액이지만, 스포티파이가 지난 분기에 2억 7000만 유로(2억 9700만 달러) 손실을 낸 것보다 개선된 것이다. [<sup>[2]</sup>](https://www.engadget.com/spotify-reaches-more-than-half-a-billion-users-for-the-first-time-142818686.html)

  - Revenue 선 그래프 (매출):
이 그래프는 분기별 매출을 나타내며, Q1 2023까지의 데이터를 포함한다. 매출은 2022년 Q1부터 2023년 Q1까지 꾸준히 증가하는 추세를 보인다. 하지만, Q4 2022와 Q1 2023 사이에 약간의 하락이 있었음을 알 수 있다.[<sup>[3]</sup>](https://newsroom.spotify.com/2023-04-25/spotify-reports-first-quarter-2023-earnings/)

  - Ad-supported Revenue 막대 그래프 (광고 매출):
이 그래프는 분기별 광고 매출을 나타낸다. Q1 2022부터 Q1 2023까지 광고 매출은 상승했지만, Q4 2022와 Q1 2023 사이에 약간의 감소가 있었다. [<sup>[3]</sup>](https://newsroom.spotify.com/2023-04-25/spotify-reports-first-quarter-2023-earnings/)

 ![c](https://github.com/5solemi5/sentiment_analysis/assets/104000117/d5310985-abcf-4d17-b0ee-a27c74fe5985)
 
 <div align=center>
  
[[자료: RYM사이트 배너]](https://rateyourmusic.com/)
  
 </div>
 
  음악 시장의 수요 증가와 더불어 발전하는 IT기술이 음악 스트리밍 플랫폼을 성장시키고 있다. 기존의 음악 스트리밍 플랫폼은 단순히 음악을 스트리밍하기 위한 플랫폼이었지만, 지금은 다양한 기능들을 제공하며 기본적으로 댓글과 평점과 같이 청취자가 실제로 앨범에 대해 피드백을 줄 수 있는 환경이 있다. 그리고 스트리밍 플랫폼 외에도 AllMusic, Pitchfork, RYM 등과 같은 음악 관련 웹사이트나 앱에서도 앨범에 대한 리뷰와 평점을 남길 수 있고 이러한 서비스들이 많아지고 있다. 따라서 음악 플랫폼의 시장이 확장되고 음악 리뷰 데이터를 얻을 수 있는 환경이 많아지는 현재, 다양한 음악 플랫폼에 많은 음악 리뷰 데이터들이 쌓이고 있다. 
   
 
## 1.2 문제정의

 <div align=center><img src = "https://github.com/5solemi5/sentiment_analysis/assets/104000117/3eee0ccc-1c2c-483d-9345-767b5d5b3c76" width="970">
  
[[자료: Cyanite사이트의 How Do AI Music Recommendation Systems Work 그림]](https://cyanite.ai/2021/09/02/how-do-ai-music-recommendation-systems-work/)
   
 </div>
 
  이미 Spotify, Apple Music, Vibe 등의 플랫폼에서 사용자가 들은 음악에 대한 정보, 재생 횟수, 스킵 여부 등의 데이터들을 이용한 AI 서비스가 배포되고 있는 상황이다. 이처럼 음악 리뷰 데이터도 AI 기술을 통해 유의미하게 이용될 수 있다. 
  
  과거에는 음악평론가들이 주로 음악 리뷰를 작성했지만, 최근에는 일반 사용자들도 자유롭게 리뷰를 작성할 수 있는 환경이 되었다. 이렇게 모인 음악 리뷰 데이터는 대중의 선호도와 음악 산업의 동향 파악에 유용한 정보를 제공한다. 이 데이터에 감성분석 모델을 적용하면, 이 데이터를 분석하여 음악 추천 시스템을 개발하거나, 음악 장르와 아티스트의 인기도 등을 파악하여 마케팅 전략을 수립할 수 있다.  
  
  이는 음악 산업에서 매우 중요한 분야가 될 수 있으며, 다양한 역량을 강화하는 데에도 도움이 된다. 때문에 본인은 첫걸음으로 음악 리뷰 데이터를 사용해서 MobileBert를 활용한 긍부정 예측 딥러닝 프로젝트를 해볼 예정이다.

# 2.데이터
## 2.1 원시 데이터 현황

- 원시 데이터 출처: Kaggle에서 제공하는 [Music Album Reviews and Ratings Dataset](https://www.kaggle.com/datasets/michaelbryantds/78k-music-album-reviews) 데이터셋을 이용한다.

- 이 데이터 세트는 2022년 5월에 [RYM사이트](https://rateyourmusic.com/)에서 스크래핑했다. 

![d](https://github.com/5solemi5/sentiment_analysis/assets/104000117/cc1037d2-4e78-400f-96a0-bcc7556a22cd)

<div align=center>
[[자료: RYM사이트 리뷰의 일부분]](https://rateyourmusic.com/)
</div>
  
- 데이터 형태:
79922여개의 Review가 있고 Rating은 부정에서 긍정을 0~5 사이로 점수를 매겼다.

|Index|Review|Rating|
|-|-|-|
|1|i think i actually under-rate ok computer...|5|
|2|when i was 15 and it was maybe the fourth...|5|
|3|atmospheric a rock anthem as the band would...|4.5|

[자료: 원시 데이터의 형태]

<div align=center><img src = "https://user-images.githubusercontent.com/104000117/235824482-9bc6893d-d1b9-4d4c-acf6-afc13a0fe025.png" width="690">

위 그래프를 통해 Rating의 긍정의 부분에 Review 수가 치우쳐 있음을 볼 수 있다.
  
</div>
  
- 데이터 부가 정보:

  - Review
  
  학습하는 Review의 너무 길거나 짧은 문장들은 딥러닝 학습에 무의미한 데이터들이므로 제거하는 과정을 거친다. 
  
  <div align=center><img src = "https://github.com/5solemi5/sentiment_analysis/assets/104000117/664e2aef-422d-4200-afe7-99607e6bcd74" width="850">
 
  [자료: 원시 데이터의 Distribution of Review Length그래프]
  
  </div>

실행 결과, 863~32759 길이의 문장들이 추출된다.


  - Rating

|Valid: 유효한 데이터 수|97% (78,200 개)|
|-|-|
|Mismatched: 일치하지 않는 데이터 수|0% (0 개)|
|Missing: 결측치 수|3% (2,084 개)|
|Mean: 평균 값|4.25|
|Std. Deviation: 표준 편차|0.87|
|Quantiles: 데이터 값의 분위수|-|
|Min|0.5|
|25%|4| 
|50%|4.5|
|75%|5|
|Max|5|

## 2.2 데이터 가공

- 임계값(threshold)

- 임계값을 기준으로 위의 데이터를 0 또는 1로 분류한 결과
  
![그림3](https://user-images.githubusercontent.com/104000117/232919132-60083ffb-0de6-443d-9b2f-f32a8d3ad646.png)

(추가/수정사항: 임계값 구한 방법 설명.....
이진분류 (출력)/파이차트, 제거 데이터, 입력 데이터에 대한 설명/문장 데이터=>길이 분포 len(str)/도수분표, 최종 데이터 (f_data, ,csv, 엑셀, 제이쓴..), raw_data.csv 가공 source.py 이력들 기록, <데이터 접근> )

# 딥러닝 모델링

# Reference

[1]https://test.hri.co.kr/upload/board/201921514759[1].hwp

[2]https://www.engadget.com/spotify-reaches-more-than-half-a-billion-users-for-the-first-time-142818686.html

[3]https://newsroom.spotify.com/2023-04-25/spotify-reports-first-quarter-2023-earnings/
