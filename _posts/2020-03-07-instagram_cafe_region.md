---
layout: post
title:  "인스타그램 크롤링 및 카페 인기지역 시각화"
subtitle:   ""
categories: data_science
tags: data_analysis
mathjax: true
comments: true
---

SNS는 트렌드의 변화를 실시간으로 파악하기 쉬운 도구이다. 인스타그램을 이용하여 카페 관련 게시글로 업로드 되는 지역을 크롤링 한 뒤, 워드클라우드를 통하여 시각화 해보았다.

# 1. 웹 크롤링

가장 먼저 해야할 일은 데이터 수집이다. 데이터 수집방법은 아래와 같이 진행하였다. 

1. '카페'와 관련된 키워드로  [#카페투어, #디저트카페, #카페그램, #카페탐방, #카페스타그램, #분위기좋은카페, #예쁜카페, #감성카페, #카페, #카페추천]이 쓰였다. 그냥 "카페"만 키워드로 썼더니, 방문자수와 맞팔을 위하여 관련없는 해시태그를 단 게시물들이 많았다. 수집 정확도를 높이기 위해 다양한 키워드를 사용하였다.
2. 카페 관련 키워드로 게시물 검색 후 url들을 추출하여 url 리스트를 만든다.
3. url 리스트로 해당 게시물에 접근하여 날짜, 댓글, 관련 해시태그들을 크롤링한다.

## 1.1 모듈 장착

```python
# basic
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# crawling
from selenium import webdriver
from selenium.webdriver.common.keys import Keys

# extra module
import time
import os
import glob
import re
import nltk
from datetime import datetime

# visualizing
import wordcloud
from PIL import Image
```

개인적으로 분석에 필요하든, 필요하지 않든 numpy, pandas, matplotlib.pyplot은 장착하고 시작한다. 웬만하면 쓰이기 때문이다. 크롤링 도구로는 selenium 모듈을 사용했고, 시각화를 위해 wordcloud 모듈을 사용하였다.

## 1.2 크롤링에 필요한 함수 생성

### 1.2.1 login 함수

```python
def instagram_login(ID, PW):
    '''
    ID랑 PW를 입력받아 인스타그램에 로그인 할 수 있게 해준다.
    '''
    driver.find_element_by_name('username').send_keys(ID)
    driver.find_element_by_name('password').send_keys(PW)
    driver.find_element_by_xpath('//*[@id="react-root"]/section/main/div/article/div/div[1]/div/form/div[4]').click()
    try:
        time.sleep(2)
        driver.find_element_by_xpath('/html/body/div[3]/div/div/div[3]/button[2]').click()
    except:
        None
```

인스타그램 웹사이트에서 서핑을 하다보면, 느닷없이 로그인창이 떠 크롤링을 방해할 때가 생겨, 아예 로그인을 하고 시작하기 위해 위와 같은 함수를 만들었다.

### 1.2.2 게시물 검색 함수

```python
def find_posts(hashtag):
    '''
    키워드를 문자열로 입력받아 키워드에 대응하는 게시물을 보여준다.
    '''
    url = 'https://www.instagram.com/explore/tags/' + hashtag + '/'
    driver.get(url)
```

검색 키워드(*ex) 카페탐방, 카페스타그램*)를 입력받아 해당 키워드에 해당하는 게시물을 검색한다.

### 1.2.3 게시물 url 추출 함수

```python
def get_urls(max_num_posts = 100000):
    '''
    해시태그에 대응하는 게시물들의 url 정보들을 가져와 list 형태로 반환한다.
    '''
    num_post = driver.find_element_by_xpath('//*[@id="react-root"]/section/main/header/div[2]/div/div[2]/span/span').text
    num_post = int(num_post.replace(',', ''))
    
    if num_post <= max_num_posts:
        max_num_posts = num_post

    body = driver.find_element_by_tag_name("body")
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    
    url_list = []
    
    while len(url_list) <= max_num_posts :
        
        for i in range(1, 9):
            post_line = driver.find_element_by_xpath('//*[@id="react-root"]/section/main/article/div[2]/div/div[' + str(i) + ']')
            a_tag_list = post_line.find_elements_by_tag_name('a')
            
            for i in range(3):
                url = a_tag_list[i].get_attribute('href')
                
                if len(url_list) < 51:
                    if url not in url_list:
                        url_list.append(url)
                    else:
                        pass
                else:
                    if url not in url_list[-50 :]:
                        url_list.append(url)
                    else:
                        pass
                    
        body.send_keys(Keys.PAGE_DOWN)            
        body.send_keys(Keys.PAGE_DOWN)
        time.sleep(1)
        body.send_keys(Keys.PAGE_UP)
        body.send_keys(Keys.PAGE_DOWN)
        time.sleep(1)
    
    return url_list
```

키워드로 검색된 게시물들의 url을 추출한다. 게시물 수는 디폴트 값을 10만으로 잡아두었는데, 1000만 해도 적당한 것 같다.

### 1.2.4 게시물 속 데이터 추출 함수

```python
def get_date():
    time_tag = driver.find_element_by_tag_name('time')
    date = time_tag.get_attribute('datetime')[:10]
    
    return date

def get_loc():
    '''
    게시물의 장소 정보를 반환한다.
    '''
    loc_list = driver.find_elements_by_xpath('//*[@id="react-root"]/section/main/div/div/article/header/div[2]/div[2]/div[2]/a').text
    
    if len(loc_list) == 0:
        
        return ''
    else:
        
        return

def get_likes():
    '''
    게시물의 좋아요 수를 반환한다.
    '''
    try:
        likes = driver.find_element_by_xpath('//*[@id="react-root"]/section/main/div/div/article/div[2]/section[2]/div/div/button/span').text
        
        return int(likes)
    except:
        
        return 0
    
def get_comments():
    '''
    게시물의 코멘트들을 반환한다.
    '''
    replies = driver.find_elements_by_class_name('EizgU')
    
    if len(replies) == 0:
        comments_class = driver.find_elements_by_class_name('C4VMK')
        comments = ''

        for i in range(len(comments_class)):
            comment =  comments_class[i].find_element_by_tag_name('span').text
            comments += comment + ' ' 
        
        return comments
    else:
        for i in range(len(replies)):
            replies[i].click()

        comments_class = driver.find_elements_by_class_name('C4VMK')
        comments = ''

        for i in range(len(comments_class)):
            comment =  comments_class[i].find_element_by_tag_name('span').text
            comments += comment + ' ' 
        
        return comments


def get_hashtag(text):
    '''
    게시물에서 반환받은 코멘트로부터 해시태그들을 추출하여 리스트로 반환한다.
    '''
    hashtag_regex = "#([0-9a-zA-Z가-힣]*)"
    hashtag_compile = re.compile(hashtag_regex)
    hashtag_list = hashtag_compile.findall(text)
    
    return hashtag_list
```

게시물에 담겨있는 날짜, 좋아요수, 댓글, 해시태그들을 추출하는 함수들이다.

### 1.2.5 데이터프레임 생성 함수

```python
def make_df(url_list):
    '''
    url list를 받아 url에 대응하는 위치, 좋아요 수, 코멘트에 관한 정보를
    데이터프레임 형태로 만들어준다.
    '''
    
    refined_url_list = []
    date_list = []
    # loc_list = []
    # likes_list = []
    comments_list = []
    hashtag_list_list = []
    
    for url in url_list:
        
        try:
            driver.get(url)
            
            # 날짜 리스트 생성
            date = get_date()
            date_list.append(date)
            
            # 위치 리스트 생성 (시간이 오래걸려 뺌)
            # loc = get_loc()
            # loc_list.append(loc)
            
            # 좋아요 수 리스트 생성 (시간이 오래걸릴 뿐더러 분석에 불필요하다고 판단하여 뺌)
            # likes = get_likes()
            # likes_list.append(likes)
            
            # 코멘트 리스트 생성
            comments = get_comments()
            comments_list.append(comments)
            
            # 해시태그 리스트 생성
            hashtag_list = get_hashtag(comments)
            hashtag_list = str(hashtag_list).replace("'", '').replace('[', '').replace(']', '')
            hashtag_list_list.append(hashtag_list)
            
            # 삭제되지 않은 게시물의 url 리스트 생성
            refined_url_list.append(url)
        
        except:
            pass
        
    df = pd.DataFrame({'url' : refined_url_list,
                       'date' : pd.to_datetime(date_list),
                       # 'loc' : loc_list,
                       # 'likes' : likes_list,
                       'comments' : comments_list,
                       'hashtag_list' : hashtag_list_list})
    
    return df
```

추출된 데이터들을 바탕으로 데이터프레임을 생성하는 함수이다.

## 1.3 크롤링

```python
# set parameter
chrome_driver_path = 'chromedriver.exe' # 크롬 드라이버 위치
#hashtag_list = ['카페투어', '디저트카페', '카페그램', '카페탐방', '카페스타그램', '분위기좋은카페', '예쁜카페', '감성카페', '카페', '카페추천'] # 추출하고 싶은 게시물안에 속한 해시태그 리스트
hashtag_list = ['카페투어', '디저트카페', '카페그램', '카페탐방', '카페스타그램', '분위기좋은카페', '예쁜카페', '감성카페', '카페', '카페추천']
ID = 'ssmoooooon' # ID
PW = '**********' # PW
num_post = 50 # 추출하고 싶은 게시물의 수
now = datetime.now()
save_date = '_' + str(now)[2:10].replace('-', '_')
# crawling start!

for hashtag in hashtag_list:

    # 드라이버 생성
    driver = webdriver.Chrome(chrome_driver_path)
    driver.get('https://www.instagram.com/accounts/login/?source=auth_switcher')
    driver.implicitly_wait(1)
    time.sleep(3)
    
    # 로그인
    instagram_login(ID, PW)
    time.sleep(3)
    
    # 해시태그 검색
    find_posts(hashtag)
    time.sleep(3)
    
    start_url_search_time = time.time()
    url_list = get_urls(num_post)
    end_url_search_time = time.time()
    
    start_make_df_time = time.time()
    df = make_df(url_list)
    end_make_df_time = time.time()
    
    print('[' + hashtag + ']')
    print('url search time : ', end_url_search_time - start_url_search_time)
    print('make df time : ', end_make_df_time - start_make_df_time)
    print()
    
    df.to_csv(hashtag + save_date + '.csv', index = False)
    driver.close()
```

```
[카페스타그램]
url search time :  15.082075595855713
make df time :  138.41524124145508

[분위기좋은카페]
url search time :  14.734349012374878
make df time :  119.3596453666687

[예쁜카페]
url search time :  19.35756516456604
make df time :  155.89054822921753

[감성카페]
url search time :  14.73903226852417
make df time :  128.90739917755127

[카페]
url search time :  14.848803281784058
make df time :  152.75569677352905

[카페추천]
url search time :  19.837244749069214
make df time :  144.49719405174255
```

위에서 만든 함수들을 이용하여 데이터 크롤링을 실행하였다. 각각의 키워드와 검색 날짜에 맞게 csv형식의 파일로 저장하였다.

```python
df.head(10)
```

![2020-03-08-df](/assets/img/2020-03-08-df.PNG)

데이터 추출이 성공적으로 된 것을 확인 할 수 있다.

# 2. 데이터 전처리

## 2.1 데이터 불러오기

```python
os.chdir('C:/Users/ssmoon/Desktop/instagram_analysis/data/raw_data')
csv_name_list = glob.glob('*')
csv_file_list = []

for csv_name in csv_name_list:
    csv_file_list.append(pd.read_csv(csv_name))
    
insta_df = pd.concat(csv_file_list)
insta_df = insta_df.drop_duplicates() # 중복행 제거
insta_df = insta_df[insta_df['hashtag_list'].notnull()] # hashtag 결측치 제거
insta_df = insta_df.reset_index()
del insta_df['index']
```

csv로 저장된 파일들을 모아놓은 폴더로 디렉토리를 변경하여 검색키워드별 csv파일들을 하나로 합쳤다.

## 2.2 지역명 추출

보통 지역이름과 카페를 붙여 카페가 위치한 지역을 나타낸다. 예를들면 연남동카페, 홍대카페 등... 이러한 특징을 이용하여 지역이름을 뽑아낼 것이다. 정규표현식 모듈을 사용하였다.

### 2.2.1 카페로 끝나는 해시태그를 찾는 함수 생성 및 적용

```python
# 카페로 끝나는 해시태그 찾는 함수 생성
def get_cafe(text):
    cafe_regex = '[0-9a-zA-Z가-힣]*카페'
    cafe_compile = re.compile(cafe_regex)
    cafe_list = cafe_compile.findall(text)
    
    return cafe_list

# 함수 적용
cafe_list = []

for i in range(len(insta_df)):
    tmp_list = get_cafe(insta_df.loc[i, 'hashtag_list'])
    cafe_list += tmp_list
    
cafe_list[1:11]
```

```
['카페', '감성카페', '카페', '갬성카페', '브런치카페', '감성카페', '제주카페', '감성카페', '대학로카페', '혜화역카페']
```

1. '카페가 들어간 해시태그들을 추출하였지만 '감성카페', '브런치카페'와 같은 지역명이 나타나지 않은 해시태그들이 보인다. 
2. '서울', '부산'과 같은 광범위한 지역인 도시이름도 보인다. 지역이 너무 넓으면 의미가 없다고 판단되었다.

이러한 해시태그들은 불용어 처리를 해주도록 하자.

### 2.2.2 '카페' 단어 제거 및 빈도수 딕셔너리 생성

```python
# '카페' 단어 제거
cafe_list_copy = cafe_list
cafe_list = []

for i in range(len(cafe_list_copy)):
    
    if len(cafe_list_copy[i]) > 2:
        cafe_list.append(cafe_list_copy[i][:-2])

# 빈도수 딕셔너리 생성
freqdist = nltk.FreqDist(cafe_list)
```

```python
freqdist.most_common(10)
```

```
[('서울', 670),
 ('감성', 661),
 ('예쁜', 608),
 ('분위기좋은', 522),
 ('부산', 342),
 ('디저트', 340),
 ('신상', 321),
 ('대구', 216),
 ('대전', 196),
 ('홈', 177)]
```

### 2.2.3 불용어 처리

![2020-03-08-stop_words](/assets/img/2020-03-08-stop_words.png)

빈도수가 높은 단어들을 추출하였고, 상위 100개의 해시태그들 중, 지역명이 아니거나, 도시와 같은 광범위한 지역 이름을 불용어 처리하였다. 불용어 처리 과정이 번거롭다고 생각할 수 있겠지만, 데이터수가 많아질수록 자잘한 불용어는 빈도수 상위권에서 걸러져 불용어 처리가 어렵지 않았다.

```python
# 불용어 파일 불러오기
stopword = open('C:/Users/ssmoon/Desktop/instagram_analysis/data/raw_data/불용어.txt',
                encoding = 'utf-8')
stopword_list = stopword.readlines()

# 불용어 전처리
for i in range(len(stopword_list)):
    stopword_list[i] = stopword_list[i].replace('\n', '')

# 도시 이름 파일 불러오기
dosi = open('C:/Users/ssmoon/Desktop/instagram_analysis/data/raw_data/도시.txt',
            encoding = 'utf-8')
dosi_list = dosi.readlines()


# 도시 이름 전처리
for i in range(len(dosi_list)):
    dosi_list[i] = dosi_list[i].replace('\n', '')

for i in range(len(dosi_list)):
    dosi_list[i] = dosi_list[i].replace('시', '')

# 불용어 리스트에 도시 이름 추가
stopword_list += dosi_list

# 빈도수 딕셔너리에서 불용어 제거
for stopword in stopword_list:
    del freqdist[stopword]
```

# 3. 워드클라우드

## 3.1 빈도수 그래프

워드클라우드 기법을 쓰기전에 단순 빈도수 그래프를 그려보았다.

```python
plt.figure(figsize = (12, 6))
freqdist.plot(20)
plt.show()
```

![2020-03-08-plot](/assets/img/2020-03-08-plot.png)

그래프를 보면 연남동과 홍대가 다른 지역에 압도적으로 많았다. 전포와 서면 또한 많았는데 서면에 전포 카페거리가 있어 빈도수가 높게 나온것으로 보인다. 우리나라는 서울이면 연남동, 부산이면 전포가 카페로 많이 찾는 지역이라고 유추할 수 있다.

단순 그래프로 확인했으니, 워드클라우드를 이용하여 시각화 효과를 높여보자.

## 3.2 워드클라우드

```python
wc = wordcloud.WordCloud(width = 1000,
                         height = 600,
                         background_color = 'white',
                         font_path = 'C:/Users/ssmoo/Downloads/SangSangFlowerRoad.otf',
                         max_words = 100)
plt.figure(figsize = (16, 10),
           dpi = 200)
plt.imshow(wc.generate_from_frequencies(freqdist))
plt.show()
```

![2020-03-08-wordcloud1](/assets/img/2020-03-08-wordcloud1.png)

- 언급 빈도수 상위 100개의 지역을 시각화하였다.
- 글씨가 클 수록 빈도수가 높다는 것을 나타낸다.

## 3.3 그림 효과를 넣은 워드클라우드

카페 분석에 맞게 커피잔 이미지를 이용하여 시각화를 재밌게 표현해 보았다.

```python
coffee_mask = np.array(Image.open('C:/Users/ssmoon/Desktop/coffee_cup.jpg'))
plt.figure(figsize = (8, 8))
plt.imshow(coffee_mask, cmap = plt.cm.gray, interpolation = 'bilinear')
```

![2020-03-08-coffe_mask](/assets/img/2020-03-08-coffe_mask.png)

위와 같은 커피 이미지를 마스크로 이용하였다.

```python
wc = wordcloud.WordCloud(width = 1000,
                         height = 600,
                         background_color = 'white',
                         font_path = 'C:/Users/ssmoo/Downloads/SangSangFlowerRoad.otf',
                         max_words = 100,
                         mask = coffee_mask)
plt.figure(figsize = (10, 10),
           dpi = 200)
plt.imshow(wc.generate_from_frequencies(freqdist))
plt.axis('off')
plt.show()
```

![2020-03-08-wordcloud2](/assets/img/2020-03-08-wordcloud2.png)

커피잔 이미지 마스크에 맞춰 워드클라우드 시각화 기법이 적용되었다.

# 4. 분석 의의

이로써 인스타그램 크롤링 및 카페 인기 지역 시각화가 완료되었다. 데이터 수집일은 19년 11월 27일 하루였지만 지속적인 크롤링을 한다면 카페 인기 지역 변화와 함께 트렌드를 예측 및 분석 할 수 있을 것이다.

이번 게시물은 SNS 웹크롤링과 워드클라우드 시각화에 초점을 두었기 때문에, 추가적인 데이터 수집은 없다. 하지만 기회가 된다면 이런 기법들을 SNS에서 제품이나 기업의 이미지를 자연어 처리하여 분석 후 마케팅에 도움을 주는 방식으로 사용해 보고싶다.