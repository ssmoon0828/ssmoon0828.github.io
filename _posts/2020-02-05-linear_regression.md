---
layout: post
title:  "선형 회귀(Linear Regression)"
subtitle:   ""
categories: data_science
tags: machine_learning
mathjax: true
comments: true
---

이 글은 **선형 회귀**를 다룬다. 더 나아가 잔차분석, 다중공선성, p-value등의 통계적 개념과 함께 **회귀 분석**을 공부하고 싶다면 [R을 이용한 회귀분석](http://www.yes24.com/Product/Goods/7182246) 책을 추천한다. 기회가 된다면 회귀분석을 주제로 포스팅 하겠다.

# 1. 선형회귀의 개념

살다보면 상관관계가 짙은 두 변수를 찾을 수 있다. 예를 들면 '키'와 '몸무게' 혹은 '중간고사 성적'과 '기말고사 성적' 등... 아래의 예시는 [kaggle](https://www.kaggle.com/mustafaali96/weight-height) 로 부터 얻은 키와 몸무게 데이터를 일부 추출하여 산점도(scatter plot)로 표현한 것이다.

![scatter_plot](/assets/img/2020-02-05-linear_regression-scatter_plot.png)

위의 예시로 부터 알 수 있듯 상관관계가 짙은 두 **연속형 변수**는 한 변수의 변화에 다른 변수가 증가하거나 감소하는 경향을 보이게 된다. 산점도속의 데이터들의 형태에서 선형성을 찾을 수 있고, 상관관계가 짙어질수록 선형성은 강해진다. 이러한 성질을 이용해 한 연속형 변수로부터 다른 연속형 변수를 예측하고자 한다. 산점도 속 데이터들의 선형을 일직선으로 표시하는 방법으로 말이다.

![scatter_plot_with_regression_line](/assets/img/2020-02-05-linear_regression-scatter_plot_with_regression_line.png)

만약 키가 170cm인 사람의 몸무게를 예측하고 싶다면 위의 예시 사진에서 그려놓은 빨간 선을 이용 할 수 있다. 이것을 **회귀선**이라 부른다. x축에서 170을 찾고 이에 대응하는 빨간 회귀선의 y축 지점을 찾으면 된다. 위의 예시에서는 대략 75 kg으로 확인된다. 키가 170cm인 모든 사람의 몸무게가 정확히 75kg은 아닐 것이다. 오차는 항상 존재할 것이다. 하지만 전체 데이터셋을 본다면 75kg은 오차가 가장 작은 예측값이 된다. 

선형회귀의 핵심은 연속형 변수(원인변수)의 값들로 부터 연속형 변수(결과변수)의 값을 예측한다는 것이다. 위의 예시에서는 '키'라는 원인변수 하나를 이용하여 '몸무게'라는 결과변수를 예측했지만*(이를 단순회귀라 부른다.)* 원인변수를 여러개 설정하여 연속형 변수 값을 예측 할 수 있다.*(이를 다중회귀라 부른다.)* 독립변수들만 잘 설정한다면 유용하게 쓰일 예측 모델을 만들 수 있을 것이다.

그렇다면 이 회귀선은 어떻게 얻을 수 있을까? 대충 어림잡아 회귀선을 긋는다면 사람마다 다른 회귀선을 그릴 것이다. 회귀선을 긋는 명확한 기준을 알아보자.

## 1-1. 아이디어

![idea](/assets/img/2020-02-05-linear_regression-idea.png)
*이미지 출처 : https://freshrimpsushi.tistory.com/548*

위의 사진을 보면 왼쪽의 연두색 회귀선은 오른쪽 빨간색 회귀선에 비해 잘 적합 되었다고 할 수 있다. 그 이유는 오차를 가장 작게 설명하고 있기 때문이다. 검은색 점에 대응되는 y값은 실제값이다. 그리고 보라색 점에 대응되는 y값은 예측값이다. 이 둘의 차이, 즉 오차는 파란색 선의 길이로 나타내어진다. 파란색 길이들이 줄어들면 줄어들수록 오차가 작아진다는 말이 된다. 오차가 가장 작은 회귀선을 찾는 것이 선형 회귀의 아이디어이다.

## 1-2. 주의점

두 변수들간 상관관계가 높을 수록 선형성이 짙어진다. 하지만 상관관계가 낮다면 선형성이 낮아지고 회귀선에 대한 설명력도 약해진다.

# 2. 수식

회귀계수를 추정해보자

## 2-1. 단순회귀

![scatter_plot_with_regression_line](/assets/img/2020-02-05-linear_regression-scatter_plot_with_regression_line.png)

오차가 가장 작은 회귀선을 찾기위해 회귀선의 $y$절편과 기울기를 찾아야한다. 회귀선은 일직선이며 일직선의 모양을 갖는 함수는 일차함수뿐이다. 중학교 1학년 시절 배웠던 일차함수식을 떠올려보자.

$$ y = ax + b $$

$a$는 $x$의 계수이자 그래프에서 기울기를 나타내었다. $b$는 상수항이며 $y$절편을 의미하였다. 회귀선도 똑같은 원리다. 회귀식은 다음과 같이 나타낼수 있다.

$$ y = \beta_{0} + \beta_{1} x $$

여기서 $\beta_{0}$는 절편을 나타내는 회귀계수가 되며 $\beta_{1}$는 $x$의 계수이자 기울기를 나타내는 회귀계수가 된다. 위의 식에 $x$값을 넣어 $y$값을 예측할 수 있다. 그러므로 데이터가 총 $n$개 있다고 할 때, $i$번째 데이터에 대한 예측값은 다음과 같이 나타낼 수 있다.

$$ \hat{y_{i}} = \beta_{0} + \beta_{1} x_{i} $$

$y$위에 그려진 $\hat{}$은 예측값을 나타낼 때 쓰이는 기호이다. 그렇다면 데이터 포인트, 즉 실제값에 대한 식은 어떻게 나타낼 수 있을까? 다음과 같다.

$$ y_{i} = \beta_{0} + \beta_{1} x_{i} + \epsilon_{i} $$

여기서 $\epsilon_{i}$ 이 추가 된 점을 주목해 볼 필요가 있다. $\epsilon_{i}$는 $i$번째 데이터에 대한 오차로 예측값과 실제값의 차이를 의미한다. 이는 다음과 같이 나타낼 수 있다.

$$ \epsilon_{i} = y_{i} - \hat{y_{i}} = y_{i} - \beta_{0} - \beta_{1} x_{i} $$

**오차들의 합**을 가장 최소로 하는 회귀계수들을 구할 것이다. 오차들을 모두 합하기 전에 예측값($\hat{y_{i}}$)과 실제값($y_{i}$)의 차이가 양수와 음수 모두 나올수 있으니 모두 양수가 나올 수 있도록 오차들을 제곱해 준 뒤 합한다. 이것을 오차제곱합이라 부른다. 통계에선 이것을 RSS(Residual sum of squares)라고도 부른다.

$$ 
\begin{aligned}
RSS & = \sum_{i = 1}^n\epsilon_{i}^2 \\ 
& = \sum_{i = 1}^n(y_{i} - \hat{y_{i}})^2 \\
& = \sum_{i = 1}^n(y_{i} - \beta_{0} - \beta_{1} x_{i})^2
\end{aligned}
$$

오차를 최소로 만드는 회귀계수($\beta_{i}$)들을 구하기 위해선 RSS($\sum_{i = 1}^n\epsilon_{i}^2$)를 미분한 값이 0이 되는 회귀계수($\beta_{i}$)들을 찾으면 된다. 각각의 회귀계수들로 편미분을 해준다.

$$
\frac{\partial RSS}{\partial \beta_{0}} = \sum_{i = 1}^n (2) (-1) (y_{i} - \beta_{0} - \beta_{1} x_{i}) = 0
$$

$$
\frac{\partial RSS}{\partial \beta_{1}} = \sum_{i = 1}^n (2) (-x_{i}) (y_{i} - \beta_{0} - \beta_{1} x_{i}) = 0
$$

위 식을 정리하면,

$$
\sum_{i = 1}^n y_{i} = n \beta_{0} + \beta_{1} \sum_{i = 1}^n x_{i}
$$

$$
\sum_{i = 1}^n x_{i} y_{i} = \beta_{0} \sum_{i = 1}^n x_{i} + \beta_{1} \sum_{i = 1}^n x_{i}^2
$$

위의 두 식중 첫번째 식의 양변에 $\sum_{i}^n x_{i}$을 곱하고 두번째 식의 양변에 $n$을 곱하여 연립방정식 소거법을 이용하면 $\beta_{1}$를 구할 수 있다.

$$ \beta_{1} = \frac{n \sum_{i}^n x_{i} y_{i} - \sum_{i}^n x_{i} \sum_{i}^n y_{i}}{n \sum_{i}^n x_{i}^2 - (\sum_{i}^n x_{i})^2} $$

위의 두 식중 첫번째 식의 양변을 n으로 나누면 $\beta_{0}$ 를 구할 수 있다.

$$ \beta_{0} = \bar{y} - \beta_{1} \bar{x} $$

여기서 $\bar{y}$와 $\bar{x}$는 $y$와 $x$의 평균을 의미한다.

지금까지 **하나**의 연속형 원인변수로 **하나**의 연속형 결과변수를 예측했다. 이를 **단순회귀분석**이라고 한다. 원인변수가 하나일때는 2차원 그래프에 나타내기도 편하고 회귀계수도 둘 뿐이라 $\beta_{0}$, $\beta_{1}$만 구하면 됐었다. 하지만 원인변수가 $n$개라면 $n$개의 원인변수에 각각 대응되는 회귀계수들도 늘어나 다음과 같은 회귀식이 나올 것이다.

$$
y_{i} = \beta_{0} + \beta_{1} x_{1i} + \beta_{2} x_{2i} + \cdots + \beta_{n} x_{ni}
$$

이와 같이 **여러개**의 원인변수로 회귀계수들을 추정하는 것을 **다중회귀분석**이라 한다. 이렇게 되면 지금까지 했던 회귀계수 추정 방법으로는 식이 너무 복잡해진다.

## 2-2. 다중회귀

선형대수를 이용하면 여러개의 원인변수가 있어도 회귀계수 추정이 편리해진다. 아래와 같이 변수들을 표현하면 $y = X_{org}\beta_{org} + \epsilon$ 로 데이터들의 실제 값들을 표현할 수 있다.

$$
y = \begin{bmatrix}
y_{1} \\
y_{2} \\
\vdots \\
y_{n}
\end{bmatrix},

X_{org} = \begin{bmatrix}
x_{11} & x_{12} & \cdots & x_{1k} \\
x_{21} & x_{22} & \cdots & x_{2k} \\
\vdots & \vdots & \ddots & \vdots \\
x_{n1} & x_{n2} & \cdots & x_{nk}
\end{bmatrix},

\beta_{org} = \begin{bmatrix}
\beta_{1} \\
\beta_{2} \\
\vdots \\
\beta_{k}
\end{bmatrix},

\epsilon = \begin{bmatrix}
\epsilon_{1} \\
\epsilon_{2} \\
\vdots \\
\epsilon_{n}
\end{bmatrix}
$$

하지만 $X_{org}$과 $\beta_{org}$으론 절편계수를 표현 할 수 없으니 $X_{org}$과 $\beta_{org}$을 다음과 같이 고쳐 $y = X\beta + \epsilon$ 식을 만든다.

$$
X_{org} = \begin{bmatrix}
x_{11} & x_{12} & \cdots & x_{1k} \\
x_{21} & x_{22} & \cdots & x_{2k} \\
\vdots & \vdots & \ddots & \vdots \\
x_{n1} & x_{n2} & \cdots & x_{nk}
\end{bmatrix}

\rightarrow

X = \begin{bmatrix}
1 & x_{11} & x_{12} & \cdots & x_{1k} \\
1 & x_{21} & x_{22} & \cdots & x_{2k} \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
1 & x_{n1} & x_{n2} & \cdots & x_{nk}
\end{bmatrix},

\beta_{org} = \begin{bmatrix}
\beta_{1} \\
\beta_{2} \\
\vdots \\
\beta_{k}
\end{bmatrix}

\rightarrow

\beta = \begin{bmatrix}
\beta_{0} \\
\beta_{1} \\
\beta_{2} \\
\vdots \\
\beta_{k}
\end{bmatrix}
$$

이제 $\epsilon$에 대한 식을 만들어 오차제곱합을 최소화하는 회귀계수벡터($\beta$)를 구한다.

$$
\begin{aligned}
RSS & = \sum_{i}^n \epsilon_{i}^2 \\ \\
& = \epsilon^T \epsilon \\ \\
& = (y - X \beta)^T (y - X \beta) \\ \\
& = y^T y - y^T X \beta - \beta^T X^T y + \beta^T X^T X \beta
\end{aligned}
$$

$RSS$를 $\beta$로 미분한다. $\frac{\partial RSS}{\partial \beta} = 0$이 되는 $\beta$를 구한다. *행렬미분에 대해 잘 모른다면 [matrix cookbook](https://www.math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf) 참조*

$$
\begin{aligned}
& \frac{\partial RSS}{\partial \beta} = 0 \\ \\
\Leftrightarrow \ & (-2) \cdot X^T y + (2) \cdot X^T X \beta = 0 \\ \\
\Leftrightarrow \ & X^T X \beta = X^T y \\
\\
& \therefore \beta = (X^T X)^{-1} (X^T y)
\end{aligned}
$$

$ \beta = (X^T X)^{-1} (X^T y) $ 을 이용하여 독립변수(원인변수)가 여러개인 다중회귀에서도 회귀계수들을 구할 수 있게된다.


# 3. 파이썬 알고리즘


## 3-1. 데이터셋

앞에서 나왔다시피 [kaggle](https://www.kaggle.com/mustafaali96/weight-height) 에서 데이터를 얻었다. 성별, 키(inches), 몸무게(lbs) 의 속성으로 10000건의 데이터가 있었고, 이 중 키, 몸무게를 cm, kg단위로 변환하여 100건의 데이터를 랜덤으로 추출하였다.

```python
data = pd.read_csv('file_path')
data['Height'] = data['Height'] * 2.54
data['Weight'] = data['Weight'] * 0.453592
del data['Gender']
data = data.values

np.random.seed(0)
random_idx = np.random.randint(0, 10000, 100)
data = data[random_idx]

scatter_plot = plt.figure(figsize = (8, 6))
plt.scatter(data[:, 0], data[:, 1], label = 'data point')
plt.xlabel('키')
plt.ylabel('몸무게')
plt.grid()
plt.legend()
scatter_plot.savefig('scatter_plot.png', dpi = 200)
```

![scatter_plot](/assets/img/2020-02-05-linear_regression-scatter_plot.png)

## 3-2. 알고리즘

[scikit-learn](https://scikit-learn.org/stable/)의 머신러닝 모듈(sklearn)은 머신러닝 모델을 클래스로 구현하여 fit 속성으로 train 데이터를 적합시키고 predict 속성으로 test 데이터에 대한 예측값을 도출한다. 이와 비슷하게 구현을 시도했고, weight 속성으로 회귀계수를 반환 가능하게 구현하였다.

```python
import numpy as np

class linear_regressor:
        
    def __init__(self, bias = True):
        '''
        편향의 유무를 설정할 수 있다.
        '''
        self.bias = bias
    
    def fit(self, X_train, y_train):
        '''
        회귀계수를 구할 수 있다.
        '''

        # 단순회귀일 경우(독립변수가 한개일 경우)를 대비하여 데이터 타입을 벡터에서 행렬로 바꾸어준다. 
        if X_train.ndim == 1:
            X_train = X_train.reshape(len(X_train), 1)
        
        # 편향이 존재할경우 일벡터인 열벡터를 추가한다.
        if self.bias == True:
            one_vector = np.ones([len(X_train), 1])
            self.X_train = np.hstack([one_vector, X_train])
        else:
            self.X_train = X_train        
        self.y_train = y_train.reshape(len(y_train), 1)
        
        mat1 = np.linalg.inv(np.dot(self.X_train.T, self.X_train))
        mat2 = np.dot(self.X_train.T, self.y_train)
        weight = np.dot(mat1, mat2)
        self.weight = weight.flatten()
        
        return weight
        
    def predict(self, X_test):
        '''
        예측값을 반환한다.
        '''
        
        if self.bias == True:
            one_vector = np.ones([len(X_test), 1])
            self.X_test = np.hstack([one_vector, X_test])
        else:
            self.X_test = X_test
        
        y_pred = np.dot(self.X_test, self.weight)
        
        return y_pred
```

## 3-3. 결과

```python
lr = linear_regressor()
lr.fit(data[:, 0], data[:, 1])
weight = lr.weight
print(weight)
```

weight는 [-169.66496107    1.44240346] 로 $\beta_{0} = -169.66496107$, $\beta_{1} = 1.44240346$ 임을 확인 할 수 있었다.

$$
\text{몸무게} = -169.66496107 + 1.44240346 * \text{키}
$$

위 식이 주어진 데이터에서 키와 몸무게를 가장 잘 설명하는 식이 된다. 

현재 예시에서는 단순회귀이므로 원인변수가 한개라 회귀계수가 2개 나오지만 다중회귀에서는 변수가 $k$개일 경우 회귀계수가 $k+1$개 나올것이다.

```python
line_x = np.arange(np.min(data[:, 0]), np.max(data[:, 0]), 0.1)
line_y = weight[0] + weight[1] * line_x

scatter_plot_with_regression_line = plt.figure(figsize = (8, 6))
plt.scatter(data[:, 0], data[:, 1], label = 'data point')
plt.plot(line_x, line_y, color = 'red', label = 'regression line')
plt.xlabel('키')
plt.ylabel('몸무게')
plt.grid()
plt.legend()
scatter_plot_with_regression_line.savefig('scatter_plot.png', dpi = 200)
```

![scatter_plot_with_regression_line](/assets/img/2020-02-05-linear_regression-scatter_plot_with_regression_line.png)
