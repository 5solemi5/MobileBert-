<div align=center>
  
# 📀음악 리뷰 감성분석📀 

MobileBert를 활용한 긍부정 예측 딥러닝 프로젝트
  
음악 리뷰에는 보통 긍정적인 리뷰가 많지만, 일부 부정적인 리뷰도 있다. 이를 이진 분류 문제로 정의하여 MobileBERT 모델을 훈련시킨다. 

<img src="https://img.shields.io/badge/PyTorch-E34F26?style=flat-square&logo=PyTorch&logoColor=white"/></a>

<img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=Python&logoColor=white"/></a>
</div>

# 1. 서론
## 1.1 음악 플랫폼 시장과 음악 리뷰 데이터의 증가

  수많은 사람들이 음악 스트리밍 플랫폼을 통해 인터넷 상에서 음악이나 비디오를 스트리밍하는 서비스를 제공받고 있다. 사용자들은 인터넷에 연결된 장치에서 음악을 듣고, 저장 및 다운로드 없이 해당 음악에 대한 액세스 권한을 얻을 수 있기 때문에 다양한 음악에 대한 접근이 쉬워졌으며, 이러한 플랫폼들은 인터넷의 보급과 함께 급속도로 성장하고 있다. 전 세계 음악 스트리밍 매출액은 '12년 7.3억 달러에서 '17년 66억 달러로 연평균 55.2% 증가하고, 영상부문에서는 전 세계 OTT(Over The Top) 서비스시장 규모가 '12년 63억 달러에서 '17년 247억 달러로 연평균 31.4% 성장하였다. [[1]](https://test.hri.co.kr/upload/board/201921514759[1].hwp) 대표적인 음악 스트리밍 플랫폼으로 Spotify가 있는데, Spotify에만 2023년 3월 31일 기준으로 5억 1,500만 명이 오디오 스트리밍 서비스를 이용하고 있고 Spotify는 거의 모든 연령대 뿐만 아니라 선진국과 개발도상국 시장 모두에서 이용자 수에 대한 큰 성장을 보였다. [[2]](https://www.engadget.com/spotify-reaches-more-than-half-a-billion-users-for-the-first-time-142818686.html) 
  
  ![음악 플랫폼](https://user-images.githubusercontent.com/104000117/235765359-4925ce6d-d705-45dc-912d-ce0f8bf8ba26.png)

  음악 시장의 수요 증가와 더불어 발전하는 IT기술이 스트리밍 플랫폼의 시장을 성장시키고 있다. 음악 스트리밍 플랫폼은 기존에는 음악을 스트리밍하기 위한 단순한 플랫폼이었지만, 지금은 다양한 기능을 제공하며 대부분의 스트리밍 플랫폼에 기본적으로 댓글과 평점과 같이 청취자가 실제로 앨범에 대해 피드백을 줄 수 있는 환경이 마련되어 있다. 그리고 스트리밍 플랫폼 외에도 AllMusic, Pitchfork, Rolling Stone, NME와 같은 음악 관련 웹사이트나 앱에서도 앨범에 대한 리뷰와 평점을 남길 수 있고 이러한 서비스들이 많아지고 있다. 이와 같이 음악 플랫폼의 시장이 확장되고 음악 리뷰 데이터를 얻을 수 있는 환경이 많아지는 현재, 다양한 음악 플랫폼에 많은 음악 리뷰 데이터들이 쌓이고 있다. 
 
## 1.2 문제정의

  이미 Spotify, Apple Music, YouTube Music, Vibe 등의 플랫폼에서 사용자가 들은 음악에 대한 정보, 재생 횟수, 스킵 여부 등의 데이터들을 이용한 음악추천 AI 서비스가 배포되고 있는 상황이다. 이처럼 음악 리뷰 데이터도 AI 기술을 통해 유의미하게 이용할 수 있다. 과거에는 음악평론가들이 주로 음악 리뷰를 작성했지만, 최근에는 일반 사용자들도 자유롭게 리뷰를 작성할 수 있는 환경이 생겨났다. 이렇게 모인 음악 리뷰 데이터는 대중의 선호도와 음악 산업의 동향 파악에 유용한 정보를 제공한다. 감성분석 모델을 적용하면, 이 데이터를 분석하여 음악 추천 시스템을 개발하거나, 음악 장르와 아티스트의 인기도 등을 파악하여 마케팅 전략을 수립할 수 있다. 또한, 감성분석 모델은 아티스트의 평가와 평판 파악에도 활용될 수 있다. 이를 통해 음악 산업에서 예측력과 시장 파악력을 높이는 데 중요한 역할을 한다. 따라서 음악 리뷰 데이터의 감성분석은 음악 산업에서 매우 중요한 분야가 될 수 있으며, 다양한 역량을 강화하는 데에도 큰 도움이 된다. 때문에 본인은 첫걸음으로 음악 리뷰 데이터를 사용해서 MobileBert를 활용한 긍부정 예측 딥러닝 프로젝트를 해볼 예정이다.

# 2.데이터
## 2.1 원시 데이터 현황

(수정사향: >총건수, 라벨의 분포(평점, 긍부정, 긍부정중), 부가정보>>파이차트 내가 데이터 시각화하기)

- 출처: https://www.kaggle.com/datasets/michaelbryantds/78k-music-album-reviews
![그림1](https://user-images.githubusercontent.com/104000117/232916528-72d8be0a-6ca6-4ce3-bc63-49e4563d659b.png)
![그림2](https://user-images.githubusercontent.com/104000117/232916534-04b41e98-89d7-4401-98c7-36d431c76512.png)

## 2.2 데이터 가공

(수정사향: 임계값 구한 방법 설명.....
이진분류 (출력)/파이차트, 제거 데이터, 입력 데이터에 대한 설명/문장 데이터=>길이 분포 len(str)/도수분표, 최종 데이터 (f_data, ,csv, 엑셀, 제이쓴..), raw_data.csv 가공 source.py 이력들 기록, <데이터 접근> )

- 임계값(threshold)을 정의하고, 이 임계값을 기준으로 위의 데이터를 0 또는 1로 분류한 결과
![그림3](https://user-images.githubusercontent.com/104000117/232919132-60083ffb-0de6-443d-9b2f-f32a8d3ad646.png)


# 딥러닝 모델링

# Reference

[1]https://test.hri.co.kr/upload/board/201921514759[1].hwp

[2]https://www.engadget.com/spotify-reaches-more-than-half-a-billion-users-for-the-first-time-142818686.html



(수정사항: 신문기사 인용 주의
보고서를 통해서 작성되었을 경우, 출처작성)
