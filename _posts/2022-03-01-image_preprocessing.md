---
layout: post
title:  "이미지 전처리"
subtitle:   ""
categories: data_science
tags: preprocessing
comments: true
---

# 패키지 불러오기


```python
import numpy as np
import cv2
from PIL import ImageFont, ImageDraw, Image
import os

os.chdir("C:/Users/ssmoo/OpenCV_Study")
cv2.__version__
```




    '4.5.4'



# OpenCV (Computer Vision)

# 1. 이미지 출력


```python
img = cv2.imread('img.jpg') # 해당 경로의 파일 읽어오기
cv2.imshow('img', img) # img 라는 이름의 창에 img 를 표시
key = cv2.waitKey() # 지정된 시간 동안 사용자 키 입력 대기
print(key) # 아스키 코드로 출력
cv2.destroyAllWindows() # 모든 창 닫기
```

    98
    

## 읽기 옵션

1. cv2.IMREAD_COLOR: 컬러 이미지, 투명 영역은 무시 (기본값)
2. cv2.IMREAD_GRAYSCALE: 흑백 이미지
3. cv2.IMREAD_UNCHANGED: 투명 영역까지 포함해서 이미지 불러오기


```python
img_color = cv2.imread('img.jpg', cv2.IMREAD_COLOR)
img_gray = cv2.imread('img.jpg', cv2.IMREAD_GRAYSCALE)
img_unchanged = cv2.imread('img.jpg', cv2.IMREAD_UNCHANGED)

cv2.imshow('img_color', img_color)
cv2.imshow('img_gray', img_gray)
cv2.imshow('img_unchanged', img_unchanged)

cv2.waitKey(0)
cv2.destroyAllWindows()
```

## Shape

이미지의 height, width, channel 정보


```python
img = cv2.imread('img.jpg')
img.shape

# (390, 620, 3) => hight : 390, width = 620, channel =3,  투명도가 들어가면 4로 바뀜
```




    (390, 640, 3)



# 2. 동영상 출력

## 동영상 파일 출력


```python
cap = cv2.VideoCapture('video.mp4')

while cap.isOpened():
    ret, frame = cap.read() # ret: 성공여부, frame: 받아온 이미지 (프레임)
    
    if not ret:
        print("더 이상 가져올 프레임이 없음.")
        break
    
    cv2.imshow('video', frame)
    
    if cv2.waitKey(500) == ord('q'): # 아스키코드로 바꿔줌
        print('사용자 입력에 의해 종료합니다.')
        break
    
cap.release() # 자원 해제
cv2.destroyAllWindows()
        
```

    사용자 입력에 의해 종료합니다.
    


```python
cap = cv2.VideoCapture(0) # 0번째 카메라 장치 (Device ID)

if not cap.isOpend(): # 카메라가 잘 열리지 않는 경우
    exit() # 프로그램 종료

```


```python
cap.isOpened()
```




    False



# 3. 도형 그리기

- OpenCV 는 RGB 순이 아닌 BGR이 디폴트


```python
# 세로 480, 가로 640, 채널 3에 해당하는 스케치북 만들기
img = np.zeros((480, 640, 3), dtype = np.uint8)
# img[:] = (255, 0, 0)
# img[:] = (0, 255, 0)
img[:] = (0, 0, 255)
# img[:] = (255, 255, 255)

cv2.imshow('img', img)
cv2.waitKey()
cv2.destroyAllWindows()
```

## 일부 영역 색칠


```python
# 세로 480, 가로 640, 채널 3에 해당하는 스케치북 만들기
img = np.zeros((480, 640, 3), dtype = np.uint8)
img[100:200, 200:300] = (255, 255, 255)

cv2.imshow('img', img)
cv2.waitKey()
cv2.destroyAllWindows()
```

## 직선

직선의 종류 (line type)

1. cv2.LINE_4: 상하좌우 4 방향으로 연결된 선
2. cv2.LINE_8: 대각선을 포함한 8 방향으로 연결된 선 (기본값)
3. cv2.LINE_AA: 부드러운 선(anti-aliasing)

![image.png](attachment:image.png)


```python
img = np.zeros((480, 640, 3), dtype = np.uint8)

COLOR = (0, 255, 255) # 색깔, BGR: Yellow
THICKNESS = 3 # 두께

cv2.line(img, (50, 100), (400, 50),
         COLOR,
         THICKNESS,
         cv2.LINE_8)
# 그릴 위치, 시작 점, 끝 점, 색깔, 두께, 선 종류

cv2.line(img, (50, 200), (400, 150),
         COLOR,
         THICKNESS,
         cv2.LINE_4)

cv2.line(img, (50, 300), (400, 250),
         COLOR,
         THICKNESS,
         cv2.LINE_AA)

cv2.imshow('img', img)
cv2.waitKey()
cv2.destroyAllWindows()
```

## 원


```python
img = np.zeros((480, 640, 3), dtype = np.uint8)

COLOR = (255, 255, 0) # 색깔, BGR: Yellow
RADIUS = 50 # 반지름
THICKNESS = 3 # 두께

# 원
cv2.circle(img, (200, 100),
           RADIUS,
           COLOR,
           THICKNESS,
           cv2.LINE_AA)
# 그릴 위치, 원의 중심, 반지름, 색깔, 두께, 선 종류

# 속이 꽉 찬 원
cv2.circle(img, (400, 100),
           RADIUS,
           COLOR,
           cv2.FILLED, # -1 로도 됨
           cv2.LINE_AA)

cv2.imshow('img', img)
cv2.waitKey()
cv2.destroyAllWindows()
```

## 사각형


```python
img = np.zeros((480, 640, 3), dtype = np.uint8)

COLOR = (0, 255, 0) # 색깔, BGR: Yellow
THICKNESS = 3 # 두께

# 사각형
cv2.rectangle(img, (100, 100), (200, 200),
              COLOR,
              THICKNESS)
# 그릴 위치, 왼쪽 위 좌표, 오른쪽 아래 좌표, 색깔, 두께

# 꽉 찬 사각형
cv2.rectangle(img, (300, 100), (400, 200),
              COLOR,
              -1)

cv2.imshow('img', img)
cv2.waitKey()
cv2.destroyAllWindows()
```

## 다각형


```python
img = np.zeros((480, 640, 3), dtype = np.uint8)

COLOR = (0, 0, 255) # 색깔, BGR: Yellow
THICKNESS = 3 # 두께 

pts1 = np.array([[100, 100], [200, 100], [100, 200]])
pts2 = np.array([[200, 100], [300, 100], [300, 200]])
pts3 = np.array([[[100, 300], [200, 300], [100, 400]], [[200, 300], [300, 300], [300, 400]]])

# cv2.polylines(img, [pts1], True, COLOR, THICKNESS, cv2.LINE_AA)
# cv2.polylines(img, [pts2], True, COLOR, THICKNESS, cv2.LINE_AA)
cv2.polylines(img, [pts1, pts2], True, COLOR, THICKNESS, cv2.LINE_AA)
cv2.fillPoly(img, pts3, COLOR, cv2.LINE_AA)


# 그릴 위치, 좌표 리스트, 열림/닫힘 여부, 색깔, 두께, 선 종류

cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

# 4. 텍스트

## OpenCV 에서 사용하는 글꼴 종류

1. cv2.FONT_HERSHEY_SIMPLEX: 보통 크기의 산 세리프(sas-serif) 글꼴
2. cv2.FONT_HERSHEY_PLAIN: 작은 크기의 산 세리프 글꼴
3. cv2.FONT_HERSHEY_SCRIPT_SIMPLEX: 필기체 스타일 글꼴
4. cv2.FONT_HERSHEY_TRIPLEX: 보통 크기의 산 세리프 글꼴
5. cv2.FONT_ITALIC: 기울임(이탤릭체)


```python
img = np.zeros((480, 640, 3), dtype = np.uint8)

COLOR = (255, 255, 255)
THICKNESS = 3
SCALE = 3

cv2.putText(img, "Seo Sangmoon", (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            SCALE,
            COLOR,
            THICKNESS)

cv2.putText(img, "Seo Sangmoon", (20, 150),
            cv2.FONT_HERSHEY_PLAIN,
            SCALE,
            COLOR,
            THICKNESS)

cv2.putText(img, "Seo Sangmoon", (20, 250),
            cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
            SCALE,
            COLOR,
            THICKNESS)

cv2.putText(img, "Seo Sangmoon", (20, 350),
            cv2.FONT_HERSHEY_TRIPLEX,
            SCALE,
            COLOR,
            THICKNESS)

cv2.putText(img, "Seo Sangmoon", (20, 450),
            cv2.FONT_HERSHEY_TRIPLEX | cv2.FONT_ITALIC,
            SCALE,
            COLOR,
            THICKNESS)

# 그릴 위치, 텍스트 내용, 시작 위치, 폰트 종류, 크기, 색깔, 두께

cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

## 한글 우회 방법


```python
def myPutText(src, text, pos, font_size, font_color):
    img_pil = Image.fromarray(src)
    draw = ImageDraw.Draw(img_pil)
    font = ImageFont.truetype('font/gulim.ttc', font_size)
    draw.text(pos, text, font = font, fill = font_color)
    
    return np.array(img_pil)

img = np.zeros((480, 640, 3), dtype = np.uint8)

FONT_SIZE = 30
COLOR = (255, 255, 255)


img = myPutText(img, '서상문', (20, 50), FONT_SIZE, COLOR)


# 그릴 위치, 텍스트 내용, 시작 위치, 폰트 종류, 크기, 색깔, 두께

cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

# 5. 파일 저장

## 이미지 저장


```python
img = cv2.imread('img.jpg', cv2.IMREAD_GRAYSCALE) # 흑백으로 이미지 불러오기

cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

result = cv2.imwrite('img_save.jpg', img)
print(result)
```

    True
    

## 저장 포맷(jpg, png)


```python
img = cv2.imread('img.jpg', cv2.IMREAD_GRAYSCALE) # 흑백으로 이미지 불러오기
result = cv2.imwrite('img_save.png', img) # png 형태로 저장
print(result)
```

    True
    

## 동영상 저장


```python
cap = cv2.VideoCapture("video.mp4")

# 코덱 정의
fourcc = cv2.VideoWriter_fourcc(*'DIVX')

# 프레임 크기, FPS
width = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS) * 2

out = cv2.VideoWriter('output_fast.avi', fourcc, fps, (width, height))
# 저장 파일명, 코덱, FPS, 크기(width, height)

while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret:
        break
    
    out.write(frame)
    cv2.imshow('video', frame)
    
    if cv2.waitKey(1) == ord('q'):
        break

out.release() # 자원해제
cap.release()
cv2.destroyAllWindows()
```


```python
codec = 'seo'
print(codec)
print(*codec)
print([codec])
print([*codec])
```

    seo
    s e o
    ['seo']
    ['s', 'e', 'o']
    

# 6. 크기 조정

## 이미지

고정 크기로 설정


```python
img = cv2.imread('img.jpg', cv2.IMREAD_GRAYSCALE)
dst = cv2.resize(img, (400, 500)) # width, height 고정 크기

cv2.imshow('img', img)
cv2.imshow('resize', dst)
cv2.waitKey()
cv2.destroyAllWindows()
```

비율로 설정


```python
img = cv2.imread('img.jpg')
dst = cv2.resize(img, None, fx = 2, fy = 2) # x, y 비율 정의

cv2.imshow('img', img)
cv2.imshow('resize', dst)
cv2.waitKey()
cv2.destroyAllWindows()
```

### 보간법

- interpolation
1. cv2.INTER_AREA: 크기 줄일 때 사용
2. cv2.INTER_CUBIC: 크기 늘릴 때 사용 (속도 느림, 퀄리티 좋음)
3. cv2.INTER_LINEAR: 크기 틀릴 때 사용 (기본값)

보간법을 적용하여 축소


```python
img = cv2.imread('img.jpg')
dst = cv2.resize(img, None, fx = 0.5, fy = 0.5, interpolation = cv2.INTER_AREA) # x, y 비율 정의

cv2.imshow('img', img)
cv2.imshow('resize', dst)
cv2.waitKey()
cv2.destroyAllWindows()
```

보간법 적용하여 확대


```python
img = cv2.imread('img.jpg')
dst1 = cv2.resize(img, None, fx = 2, fy = 2, interpolation = cv2.INTER_CUBIC) # x, y 비율 정의
dst2 = cv2.resize(img, None, fx = 2, fy = 2, interpolation = cv2.INTER_LINEAR) # x, y 비율 정의

cv2.imshow('img', img)
cv2.imshow('resize1', dst1)
cv2.imshow('resize2', dst2)
cv2.waitKey()
cv2.destroyAllWindows()
```

## 동영상

고정 크기로 설정


```python
cap = cv2.VideoCapture('video.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret:
        break
    
    frame_resized = cv2.resize(frame, None, fx = 2, fy = 2, interpolation = cv2.INTER_CUBIC)
    
    cv2.imshow('video', frame_resized)
    
    if cv2.waitKey(1) == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()
```

# 7. 이미지 자르기

영역을 잘라서 새로운 윈도우(창)에 표시

img[세로범위, 가로범위]


```python
img = cv2.imread('img.jpg')
img.shape
```




    (390, 640, 3)




```python
crop = img[100:200, 200:400, :]

cv2.imshow('img', img)
cv2.imshow('corp', crop)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

영역을 잘라서 기존 윈도우에 표시


```python
crop = img[100:200, 200:400, :]
img[100:200, 400:600] = crop

cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

# 8. 이미지 대칭

## 좌우 대칭


```python
img = cv2.imread('img.jpg')
flip_horizontal = cv2.flip(img, 1) # flipCode > 0 : 좌우 대칭 Horizontal

cv2.imshow('img', img)
cv2.imshow('flip_horizontal', flip_horizontal)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

## 상하 대칭


```python
img = cv2.imread('img.jpg')
flip_vertical = cv2.flip(img, 0) # flipCode == 0 : 상하 대칭 Vertical

cv2.imshow('img', img)
cv2.imshow('flip_vertical', flip_vertical)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

## 상하좌우 대칭


```python
img = cv2.imread('img.jpg')
flip_both = cv2.flip(img, -1) # flipCode < 0 : 상하 대칭 Vertical

cv2.imshow('img', img)
cv2.imshow('flip_both', flip_both)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

# 9. 이미지 회전

## 시계 방향 90도 회전


```python
img = cv2.imread('img.jpg')
rotate_90 = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

cv2.imshow('img', img)
cv2.imshow('rotate_90', rotate_90)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

## 180도 회전


```python
img = cv2.imread('img.jpg')
rotate_180 = cv2.rotate(img, cv2.ROTATE_180)

cv2.imshow('img', img)
cv2.imshow('rotate_180', rotate_180)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

## 시계 반대 방향 90도 회전 (시계 방향 270도 회전)


```python
img = cv2.imread('img.jpg')
rotate_270 = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

cv2.imshow('img', img)
cv2.imshow('rotate_270', rotate_270)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

# 10. 이미지 변형 (흑백)

이미지를 흑백으로 불러오기


```python
img = cv2.imread('img.jpg', cv2.IMREAD_GRAYSCALE)

cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

불러온 이미지를 흑백으로 변경


```python
img = cv2.imread('img.jpg')

dst = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.imshow('img', img)
cv2.imshow('gray', dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

# 11. 이미지 변형 (흐림)

## 가우시안 블러

커널 사이즈 변화에 따른 흐림


```python
img = cv2.imread('img.jpg')

# (3, 3), (5, 5), (7, 7)
kernel_3 = cv2.GaussianBlur(img, (3, 3), 0)
kernel_5 = cv2.GaussianBlur(img, (5, 5), 0)
kernel_7 = cv2.GaussianBlur(img, (7, 7), 0)

cv2.imshow('img', img)
cv2.imshow('kernel_3', kernel_3)
cv2.imshow('kernel_5', kernel_5)
cv2.imshow('kernel_7', kernel_7)

cv2.waitKey(0)
cv2.destroyAllWindows()
```

표준 편차 변화에 따른 흐림


```python
img = cv2.imread('img.jpg')

simga_1 = cv2.GaussianBlur(img, (0, 0), 1) # sigmaX - 가우시안 커널의 x 방향의 표준 편차
simga_2 = cv2.GaussianBlur(img, (0, 0), 2) 
simga_3 = cv2.GaussianBlur(img, (0, 0), 3)

cv2.imshow('img', img)
cv2.imshow('simga_1', simga_1)
cv2.imshow('simga_2', simga_2)
cv2.imshow('simga_3', simga_3)

cv2.waitKey(0)
cv2.destroyAllWindows()
```

# 12. 이미지 변형 (원근)

## 사다리꼴 이미지 펼치기


```python
img = cv2.imread('newspaper.jpg')

width, height = 620, 240

src = np.array([[511, 352], [1008, 345], [1122, 584], [455, 594]], dtype = np.float32) # input 4개 지정
dst = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype = np.float32) # ouput 4개 지점
# 좌상, 우상 ,우하, 좌하 (시계 방향으로 4 지점 정의)

matrix = cv2.getPerspectiveTransform(src, dst)
result = cv2.warpPerspective(img, matrix, (width, height))

cv2.imshow('img', img)
cv2.imshow('result', result)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

## 회전된 이미지 펼치기


```python
img = cv2.imread('poker.jpg')

width, height = 530, 710

src = np.array([[701, 141], [1131, 417], [725, 1007], [279, 697]], dtype = np.float32) # input 4개 지정
dst = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype = np.float32) # ouput 4개 지점
# 좌상, 우상 ,우하, 좌하 (시계 방향으로 4 지점 정의)

matrix = cv2.getPerspectiveTransform(src, dst)
result = cv2.warpPerspective(img, matrix, (width, height))

cv2.imshow('img', img)
cv2.imshow('result', result)
cv2.waitKey(0)
cv2.destroyAllWindows()
```
