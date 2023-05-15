<div align=center>

![music_review](https://github.com/5solemi5/sentiment_analysis/assets/104000117/b26daeca-d549-4658-9e1a-ce33e40c15ff)
  
# 📀음악 리뷰 감성분석📀 
  
**MobileBert를 활용한 긍부정 예측 딥러닝 프로젝트**
  
음악 리뷰에는 보통 긍정적인 리뷰가 많지만, 일부 부정적인 리뷰도 있다. 이를 이진 분류 문제로 정의하여 MobileBERT 모델을 훈련시킨다. 
  
<h2>:heavy_check_mark:Tech Stack</h2>
<img src="https://img.shields.io/badge/PyTorch-E34F26?style=flat-square&logo=PyTorch&logoColor=white"/></a> 
<img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=Python&logoColor=white"/></a>

</div>

# 1. 서론
## 1.1 음악 플랫폼 시장과 음악 리뷰 데이터의 증가

  수많은 사람들이 음악 스트리밍 플랫폼을 통해 인터넷 상에서 음악이나 비디오를 스트리밍하는 서비스를 제공받고 있다. 사용자들은 인터넷에 연결된 장치에서 음악을 듣고, 저장 및 다운로드 없이 해당 음악에 대한 액세스 권한을 얻을 수 있기 때문에 다양한 음악에 대한 접근이 쉬워졌으며, 이러한 플랫폼들은 인터넷의 보급과 함께 급속도로 성장하고 있다. 전 세계 음악 스트리밍 매출액은 '12년 7.3억 달러에서 '17년 66억 달러로 연평균 55.2% 증가하였다. [[1]](https://test.hri.co.kr/upload/board/201921514759[1].hwp) 대표적인 음악 스트리밍 플랫폼으로 Spotify가 있는데, Spotify에만 2023년 3월 31일 기준으로 5억 1,500만 명이 오디오 스트리밍 서비스를 이용하고 있고 Spotify는 거의 모든 연령대 뿐만 아니라 선진국과 개발도상국 시장 모두에서 이용자 수에 대한 큰 성장을 보였다. [[2]](https://www.engadget.com/spotify-reaches-more-than-half-a-billion-users-for-the-first-time-142818686.html) 
  
![final](https://github.com/5solemi5/sentiment_analysis/assets/104000117/4ab9b803-a2cd-488d-a487-e73dedcb57a6)
![b](https://github.com/5solemi5/sentiment_analysis/assets/104000117/1cfe1464-55d2-4251-b5b9-e06d8a15357b)
 
  음악 시장의 수요 증가와 더불어 발전하는 IT기술이 음악 스트리밍 플랫폼을 성장시키고 있다. 음악 스트리밍 플랫폼은 기존에는 음악을 스트리밍하기 위한 단순한 플랫폼이었지만, 지금은 다양한 기능들을 제공하며 대부분의 스트리밍 플랫폼은 기본적으로 댓글과 평점과 같이 청취자가 실제로 앨범에 대해 피드백을 줄 수 있는 환경이 있다. 그리고 스트리밍 플랫폼 외에도 AllMusic, Pitchfork, Rolling Stone, RYM과 같은 음악 관련 웹사이트나 앱에서도 앨범에 대한 리뷰와 평점을 남길 수 있고 이러한 서비스들도 많아지고 있다. 음악 플랫폼의 시장이 확장되고 음악 리뷰 데이터를 얻을 수 있는 환경이 많아지는 현재, 다양한 음악 플랫폼에 많은 음악 리뷰 데이터들이 쌓이고 있다. 
  
  ![c](https://github.com/5solemi5/sentiment_analysis/assets/104000117/d5310985-abcf-4d17-b0ee-a27c74fe5985)
<RYM사이트 배너>
 
## 1.2 문제정의

  이미 Spotify, Apple Music, YouTube Music, Vibe 등의 플랫폼에서 사용자가 들은 음악에 대한 정보, 재생 횟수, 스킵 여부 등의 데이터들을 이용한 음악추천 AI 서비스가 배포되고 있는 상황이다. 이처럼 음악 리뷰 데이터도 AI 기술을 통해 유의미하게 이용될 수 있다. 과거에는 음악평론가들이 주로 음악 리뷰를 작성했지만, 최근에는 일반 사용자들도 자유롭게 리뷰를 작성할 수 있는 환경이 생겨났다. 이렇게 모인 음악 리뷰 데이터는 대중의 선호도와 음악 산업의 동향 파악에 유용한 정보를 제공한다. 감성분석 모델을 적용하면, 이 데이터를 분석하여 음악 추천 시스템을 개발하거나, 음악 장르와 아티스트의 인기도 등을 파악하여 마케팅 전략을 수립할 수 있다. 또한, 감성분석 모델은 아티스트의 평가와 평판 파악에도 활용될 수 있다. 
  
  이를 통해 음악 산업에서 예측력과 시장 파악력을 높이는 데 중요한 역할을 하는 음악 리뷰 데이터의 감성분석은 음악 산업에서 매우 중요한 분야가 될 수 있으며, 다양한 역량을 강화하는 데에도 큰 도움이 된다. 때문에 본인은 첫걸음으로 음악 리뷰 데이터를 사용해서 MobileBert를 활용한 긍부정 예측 딥러닝 프로젝트를 해볼 예정이다.

# 2.데이터
## 2.1 원시 데이터 현황

- 출처: [Music Album Reviews and Ratings Dataset](https://www.kaggle.com/datasets/michaelbryantds/78k-music-album-reviews)

- 이 데이터 세트는 2022년 5월에 [RYM 사이트](https://rateyourmusic.com/)에서 스크래핑했다. 

![d](https://github.com/5solemi5/sentiment_analysis/assets/104000117/cc1037d2-4e78-400f-96a0-bcc7556a22cd)
<RYM사이트 리뷰의 일부분>
  
- 데이터 형태:
79922여개의 Review가 있고 Rating은 부정에서 긍정을 0~5 사이로 점수를 매겼다.

|Index|Review|Rating|
|-|-|-|
|1|i think i actually under-rate ok computer...|5|
|2|when i was 15 and it was maybe the fourth...|5|
|3|atmospheric a rock anthem as the band would...|4.5|

![raw막대](https://user-images.githubusercontent.com/104000117/235824482-9bc6893d-d1b9-4d4c-acf6-afc13a0fe025.png)

위 그래프를 통해 Rating의 긍정의 부분에 Review 수가 치우쳐 있음을 볼 수 있다.
  
- 데이터 부가 정보:

  - Review
  
  Review의 너무 짧은 문장들은 딥러닝 학습에 무의미한 데이터들이므로 제거하는 과정을 거친다. 
  
![히](https://user-images.githubusercontent.com/104000117/235825477-a3fbbb08-9315-419f-8ea2-5ae1368bc279.png)
<Review 문장 길이 히스토그램>

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
