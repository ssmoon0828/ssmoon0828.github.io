---
layout: post
title:  "로지스틱 회귀(Logistic Regression)"
subtitle:   ""
categories: data_science
tags: machine_learning
mathjax: true
comments: true
---

로지스틱 회귀는 이름만 회귀일뿐 **분류**를 위한 머신러닝 알고리즘이다. 또한 시그모이드 함수의 개념과 함께 딥러닝(Deep Learning)의 기초가 되기도 한다. 이번 포스팅은 [ratsgo 블로그](https://ratsgo.github.io/machine%20learning/2017/04/02/logistic/) 와 [논문](https://czep.net/stat/mlelr.pdf)을 참고하였다.

# 1. 이진 로지스틱 회귀

선형회귀, 다중회귀, 다항회귀등... '회귀' 단어가 붙는 머신러닝 알고리즘은 연속형 변수로부터 연속형 변수를 예측한다. 하지만 언제나 종속변수가 연속형 변수로 존재하진 않다. 합격과 불합격, 질병의 양성판정과 음성판정, Yes or No 등... 예측해야 하는 변수가 두가지 범주를 가지는 범주형 변수일 경우도 많다. 보통 이런 경우 두가지의 범주를 '0'과 '1'로 인코딩하게 된다. 아래의 예를 보자.

![2020-03-01-lr_plot](/assets/img/2020-03-01-lr_plot.png)

공부시간에 따른 합격여부를 성공할 경우 '1', 실패할 경우 '0'으로 나타내어 산점도로 시각화 하였다. 이제 선형회귀를 이용하여 데이터들의 추세선을 그려보자.

![2020-03-01-lr_plot_with_line](/assets/img/2020-03-01-lr_plot_with_line.png)

위 그래프의 빨간 회귀선으로 '0'과 '1'만을 종속변수로 가지는 데이터들을 예측하기엔 어색하다. 예를 들어 공부시간량이 40일때는 합격여부를 1.7로 예측할 것이고, 공부시간량이 0시간인 경우에는 합격여부를 -0.2로 예측할 것이다. 심지어 어떤 학생이 공부를 너무 많이 해서 공부를 2000시간 했다고 하면 예측값은 97일 것이다. 성공 or 실패인 '1'과 '0'만을 예측값으로 갖고싶지만 선형회귀로 두 범주를 예측하기엔 예측값이 연속형 변수이기 때문에 무리가 있다.

선형회귀는 치역을 $(-\infty, \infty)$ 범위로 갖지만, 로지스틱 회귀는 치역을 $ \\{ 0, 1 \\} $의 범위에서 갖게된다. 두 범주만 예측이 가능한 로지스틱 회귀는 어떻게 모델링 할 수 있을까. 이는 **시그모이드 함수**를 이용하여 가능하게 된다.

## 1.1 승산(Odds)

시그모이드 함수를 알기 전 **승산(Odds)** 의 개념을 짚고 넘어가야 한다. 승산이란, 임의의 사건 $X$에 대하여 사건 $X$가 일어나지 않을 확률 대비 사건 $X$가 일어날 확률을 뜻하는 개념이다. 수식으로 나타내면 다음과 같이 나타낼 수 있다.

$$
Odds = \frac{P(X)}{P(X^{c})}
$$

## 1.2 시그모이드 함수

독립변수 $x = [x_{1}, x_{2}, \cdots, x_{k}]$가 주어질 때 종속변수 $y$가 0 혹은 1일 확률은 다음과 같이 나타낼 수 있다.

$$
\begin{aligned}
P(Y = 0 | X = x) \\ \\
P(Y = 1 | X = x)
\end{aligned}
$$

이 때 $y$ 는 0 혹은 1의 값 밖에 가질 수 없으므로

$$
\begin{aligned}
P(Y = 0 | X = x) + P(Y = 1 | X = x) = 1 \\ \\
P(Y = 0 | X = x) = 1 - P(Y = 1 | X = x) \\ \\
P(Y = 1 | X = x) = 1 - P(Y = 0 | X = x)
\end{aligned}
$$

위와 같은 식도 성립한다.

이제 $ P(Y = 1 \| X = x)$ 에 초점을 맞추어 독립변수 $x$가 주어질 때 선형회귀식 $X\beta$를 통하여 $y = 1$일 확률을 나타낼 것이다.

$$
\begin{aligned}
P(Y = 1 | X = x) &= x^T \beta \\
& = \beta_{0} + \beta_{1} x_{1} + \beta_{2} x_{2} + \cdots + \beta_{k} x_{k}
\end{aligned}
$$

하지만 $P(Y = 1 \| X = x)$는 0부터 1사이의 값을 갖고, $x^T \beta$는 음의무한대($-\infty$) 부터 양의 무한대($\infty$) 사이의 값을 갖기 때문에 양변의 범위가 맞지 않다.

이 때 승산의 개념을 도입하여 위의 식을 다음과 같이 변형한다.

$$
Odds = \frac{P(Y = 1 | X = x)}{1 - P(Y = 1 | X = x)} = \frac{P(Y = 1 | X = x)}{P(Y = 0 | X = x)} = x^T \beta
$$

이렇게 되면 좌변의 범위는 $P(Y = 1 \| X = x) \in (0, 1)$ 이지만 0부터 양의 무한대 $(0, \infty)$의 범위를 갖게 된다. 그래프로 나타내면 다음과 같다.

![2020-03-01-odds](/assets/img/2020-03-01-odds.png)

하지만 아직 음의무한대($-\infty$) 부터 양의 무한대($\infty$)까지의 범위가 아니므로 $\log$를 씌워주어 $ (-\infty, \infty) $의 범위를 갖게 해주면 양변의 범위가 같아지게 된다.

$$
\log \frac{P(Y = 1 | X = x)}{1 - P(Y = 1 | X = x)} \in (-\infty, \infty) = x^{T} \beta \in (-\infty, \infty)
$$

![2020-03-01-log_odds](/assets/img/2020-03-01-log_odds.png)

최종적으로 알고싶은 것은 결국 $P(Y = 1 \| X = x)$ 이다. $P(Y = 1 \| X = x)$를 $p$로, $ x^{T} \beta $를 $x$로 치환하여 $p$에 대한 식을 유도해 보자.

$$
\begin{aligned}
& x = \log \frac{p}{1 - p} \\ \\
\Leftrightarrow \ & e^{x} = \frac{p}{1 - p} \\ \\
\Leftrightarrow \ & e^{x}(1 - p) = p \\ \\
\Leftrightarrow \ & e^{x} - x e^{p} = p \\ \\
\Leftrightarrow \ & e^{x} = p + x e^{p} \\ \\
\Leftrightarrow \ & e^{x} = p(1 + e^{x}) \\ \\
\Leftrightarrow \ & p = \frac{e^{x}}{1 + e^{x}} = \frac{1}{1 + e^{-x}}
\end{aligned}
$$

$$
\therefore P(Y = 1 | X = x) = \frac{1}{1 + e^{-x^{T} \beta}}
$$

이렇게 유도된 시그모이드 함수($\sigma(x)$)는 아래와 같은 S자형 곡선의 그래프를 가지며, 다음과 같은 성질을 가진다.

![2020-03-01-sigmoid](/assets/img/2020-03-01-sigmoid.png)

- $ 0 < \sigma(x) < 1 $
- $ \sigma(0) = 0.5$

$P(Y = 1 \| X = x)$과 $P(Y = 0 \| X = x)$의 대소관계를 비교하여 확률이 더 큰 쪽을 예측값으로 결정하게 된다. $P(Y = 1 \| X = x) > (Y = 0 \| X = x)$ 인 경우를 생각해보자.

$$
\begin{aligned}
& P(Y = 1 | X = x) > (Y = 0 | X = x) \\ \\
\Leftrightarrow \ & \frac{P(Y = 1 | X = x)}{P(Y = 0 | X = x)} > 1 \\ \\
\Leftrightarrow \ & \log \frac{P(Y = 1 | X = x)}{P(Y = 0 | X = x)} > 0 \\ \\
\Leftrightarrow \ & x^{T} \beta > 0 \\ \\
\Leftrightarrow \ & \sigma(x^{T} \beta) > 0.5
\end{aligned}
$$

위와 같은 이유로 보통 $ x^{T} \beta $ 값이 0보다 큰 경우, 즉 시그모이드 함수로 부터 도출된 값이 0.5보다 크게 되면 1로 판단하게 된다. 반대로 시그모이드 함수로 부터 도출된 값이 0.5보다 작게 되면 0으로 판단하게 된다.

하지만 이것이 절대적이지는 않다. 0.5라는 기준점을 로지스틱 회귀에서는 절단값(threshold or cutoff)으로 부르게 되는데 이를 조절하여 예측 민감도를 조절 할 수 있게 된다.

![2020-03-01-sigmoid_with_cutoff](/assets/img/2020-03-01-sigmoid_with_cutoff.png)

예를 들어 병원에서 어떤 질병의 양성여부를 진단할 때는 정확도(accuracy)보다 거짓음성(false-negative) 비율이 중요할 수 있다. 질병의 양성판정의 가능성을 더 열어두는게 실제 양성판정 환자들을 잘 찾아낼 수 있고, 그것이 중요하기 때문이다. 이런 경우 양성판정에 대한 예측 값을 1로 둘 때, 절단값을 0.5로 두지 않고 그 보다 아래로 두기도 한다.

그렇다면 로지스틱 회귀의 파라메터인 계수벡터 $\beta$는 어떻게 구할 수 있을까. 난 처음에는 단순히 OLS(Ordinary Least Squares)를 이용하여 $ \beta = (X^T X)^{-1} (X^T y) $로 구해지는 줄 알았다. 하지만 이 방식으로 파이썬 모듈을 만들어 보니 예측력이 좋지 않아, 이와 관련된 글들을 더 찾아보았다.  

결국 **최대우도추정법**을 이용하여 계수벡터 $\beta$를 구해야 한다는 것을 알았고, [논문](https://czep.net/stat/mlelr.pdf)으로 부터 유도식을 얻을 수 있었다. 다만 논문에선 이항분포를 이용하여 우도와 우도의 미분값을 유도하였는데, 내 생각엔 베르누이 분포를 쓰는게 맞는 것 같았고, 아래 **4. 파이썬 코드**절에서 나오는 코드는 베르누이 분포로 유도한 식으로 코드를 짠 것이다.

# 2. 이진 로지스틱 회귀 계수벡터 추정

로지스틱회귀에서 최적의 회귀계수벡터는 어떻게 구할 수 있을까.

## 2.1 최대 우도 추정법(MLE : Maximum Likelihood Estimation)

**최대우도추정법**은 모수가 알려지지 않은 확률 분포에서 모수를 추정하는 통계기법으로 **우도**값을 가장 크게 하는 모수를 최적의 모수로 간주한다. 이는 머신러닝 모델에서 최적의 하이퍼파라메터를 찾는 방법으로 응용 될 수 있다. **우도**와 **최대우도추정법**에 대한 글은 [이곳]에 포스팅 해두었다.

로지스틱 회귀는 종속변수로 0과 1 두가지를 가지기 때문에 베르누이 분포를 전제로 한다. 베르누이 확률변수 $Y$의 분포에 대한 수식은 아래와 같다. 여기서 $p$는 $y_{i}$가 성공(1)일 확률이다. 마찬가지로 $1-p$는 $y_{i}$가 실패(0)일 확률이 된다.

$$
P(Y = y_{i}) = p^{y_{i}} (1 - p)^{1 - y_{i}} \ \ \ (y_{i} = 0, 1)
$$

이를 로지스틱 회귀에 적용하여, 계수벡터 $\beta$가 주어질때, 종속변수 $y = [y_{1}, y_{2}, \cdots, y_{n}]$에 대한 결합확률분포는 다음과 같이 나타낼 수 있다.

$$
P(y | \beta) = \prod_{i = 1}^{n} \sigma(x_{i}^{T} \beta)^{y_{i}} (1 - \sigma(x_{i}^{T} \beta))^{1 - y_{i}}
$$

이에 따라 우도함수(Likelihood function)는 다음과 같이 나타내어진다.

$$
L(\beta | y) = \prod_{i = 1}^{n} \sigma(x_{i}^{T} \beta)^{y_{i}} (1 - \sigma(x_{i}^{T} \beta))^{1 - y_{i}}
$$

이제 **최대우도추정법**을 이용하여 위의 우도함수의 최대값을 갖는 계수벡터 $\beta$를 추정한다. 먼저 우도함수를 나타내는 식의 양변에 $\log$를 취하는데, 이유는 다음과 같다.

- 로그함수를 취해도 대소관계는 유지된다. 즉 우도함수를 최대로 만드는 계수벡터는 로그함수에 영향을 받지 않는다.
- 로그함수를 이용하여 $\prod_{i = 1}^{n}$를 $\sum_{i = 1}^{n}$로 바꾸어 연산을 간편하게 만든다.

이러한 성질 덕분에 우도함수에 로그를 취한 **로그우도함수**를 많이 이용하기도 한다.

양변에 로그를 취한 우도함수는 다음과 같이 나타내어진다.

$$
\begin{aligned}
\log L(\beta | y) & = \log \prod_{i = 1}^{n} \sigma(x_{i}^{T} \beta)^{y_{i}} (1 - \sigma(x_{i}^{T} \beta))^{1 - y_{i}} \\ \\
& = \log \prod_{i = 1}^{n}(\frac{\sigma(x_{i}^{T} \beta)}{1 - \sigma(x_{i}^{T} \beta)})^{y_{i}} (1 - \sigma(x_{i}^{T} \beta)) \\ \\ 
& = \sum_{i = 1}^{n} \log (e^{x_{i}^{T} \beta})^{y_{i}} + \log (1 - \sigma(x_{i}^{T} \beta)) \\ \\
& = \sum_{i = 1}^{n} y_{i} (x_{i}^{T} \beta) - \log (\frac{1}{1 + e^{x_{i}^{T} \beta}}) \\ \\
& = \sum_{i = 1}^{n} y_{i} (x_{i}^{T} \beta) - \log (1 + e^{x_{i}^{T} \beta})
\end{aligned}
$$

$$
\therefore \log L(\beta | y, X) = y \circledcirc X \beta - \log(1 + e^{X \beta}) \ \ \ \circledcirc : \text{요소곱}
$$

이렇게 구해진 로그우도함수값을 최대로 만드는 계수벡터 $\beta$가 로지스틱회귀의 하이퍼파라메터가 된다. 로그우도함수가 비선형이기 때문에, 로그우도함수값을 최대로 만드는 계수벡터 $\beta$는 아쉽게도 명시적인 해를 구할수가 없다. 때문에 **경사상승법**을 통하여 계수벡터 $\beta$를 구하여야한다.

**경사상승법**이란 함수를 미분하여 얻은 기울기 방향으로 입력변수값을 갱신하여, 함수값을 점차 높여 함수의 최대값에 접근하는 방법이다. 사실 딥러닝에서 손실함수의 최소값을 찾는데 쓰이는 **경사하강법**이 더 유명하지만 기울기의 부호만 다르게 연산할 뿐, 본질적으로 같은 최적해 탐색법이다.

**경사상승법**을 이용하여 우도함수의 최대값을 구하기 위해 우도함수를 미분하여야한다.

## 2.2 우도 함수 미분

앞서 나온 우도함수를 미분하여 나온 계수벡터 $\beta$의 기울기 벡터는 다음과 같이 나타낼 수 있다.

$$
\frac{\partial \log L(\beta | y, X)}{\partial \beta} = X^{T} y - X^{T} \cdot \sigma(X \beta)
$$

사실 우도함수의 미분과정도 올리고 싶었으나, 귀찮다는 핑계로 생략한다. 미분과정과 결과식은 [논문](https://czep.net/stat/mlelr.pdf)에 나와있으며, 결과식은 행렬로 재표현하였다.

이제 계수벡터 $\beta$를 아래와 같이 갱신하게 되면 로그우도함수값을 최대로 만드는 계수벡터 $\beta$에 가까워지게되고, 일정수준이 되면 갱신을 멈춘 후 계수벡터 $\beta$를 로지스틱회귀의 하이퍼파라메터로 쓰면 된다.

$$
\beta \gets \beta + \lambda \frac{\partial \log L(\beta | y, X)}{\partial \beta}
$$

여기서 $\lambda$는 보통 0.01로 잡는다. $\lambda$이 너무 크면 값이 발산하게 되거나, 너무 작으면 학습률이 떨어진다.

# 3. 파이썬 코드

로지스틱 회귀 모델을 파이썬 코드로 구현해 보았다. fit 속성으로 훈련데이터를 적합시키고, predict 속성으로 특성을 예측하게 된다. threshold 매개변수로 절단값을 조절할 수 있도록 하였다.

```python
import numpy as np

class logistic_regressor:
    
    def __init__(self, bias = True, threshold = 0.5, alpha = 0.01, count = 100):
        # 절단값, 학습률, 최소학습횟수를 설정할 수 있다.
        self.bias = bias
        self.threshold = threshold
        self.alpha = alpha
        self.count = count
    
    def fit(self, X_train, y_train):
        
        if self.bias == True:
            X_train = np.hstack([np.ones([len(X_train), 1]), X_train])
        
        # 계수벡터 초기값
        weight = np.dot(np.linalg.inv(np.dot(X_train.T, X_train)), np.dot(X_train.T, y_train))
        
        # 우도
        # likelihood = np.sum(y_train * np.dot(X_train, weight) - np.log(1 + np.exp(np.dot(X_train, weight))))
        
        # 우도 미분값
        d_likelihood = np.dot(X_train.T, y_train) - np.dot(X_train.T, 1 / (1 + np.exp(-np.dot(X_train, weight))))
           
        count = 0
        
        while True:
        
            if (count <= self.count):
                weight += (self.alpha * d_likelihood)
                # likelihood = np.sum(y_train * np.dot(X_train, weight) - np.log(1 + np.exp(np.dot(X_train, weight))))
                d_likelihood = np.dot(X_train.T, y_train) - np.dot(X_train.T, 1 / (1 + np.exp(-np.dot(X_train, weight))))
                
            else:
                weight += (self.alpha * d_likelihood) # 계수벡터 갱신
                # likelihood = np.sum(y_train * np.dot(X_train, weight) - np.log(1 + np.exp(np.dot(X_train, weight))))
                post_d_likelihood = d_likelihood
                d_likelihood = np.dot(X_train.T, y_train) - np.dot(X_train.T, 1 / (1 + np.exp(-np.dot(X_train, weight))))
                    
                if (np.sum(np.abs(d_likelihood - post_d_likelihood)) < 1e-5):
                    
                    # 우도 미분값 갱신이 일정수준 이하면 멈춤
                    break
            
            count += 1
        
        self.X_train = X_train
        self.y_train = y_train
        self.weight = weight
                    
    def predict(self, X_test):
        
        if self.bias == True:
            one_vector = np.ones([len(X_test), 1])
            self.X_test = np.hstack([one_vector, X_test])
        else:
            self.X_test = X_test
        
        sigmoid = lambda x : 1 / (1 + np.exp(-x))   
        y_sigmoid_pred = sigmoid(np.dot(self.X_test, self.weight))
        y_boolean_pred = y_sigmoid_pred > self.threshold
        y_pred = y_boolean_pred.astype(np.int)
        self.y_pred = y_pred
        
        return y_pred
    
    def score(self, X_test, y_test):
        
        return np.mean(self.predict(X_test) == y_test)
```

로지스틱 회귀 모델의 예측과정을 시각적으로 나타내기 위해 sklearn 모듈의 make_blobs 데이터셋을 가져왔다.

```python
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
from sklearn.model_selection import train_test_split

X, y = make_blobs(n_samples = 200, centers = 2, n_features=2, random_state=0, cluster_std = 0.4)

plt.scatter(X[:, 0], X[:, 1], c = y, cmap = 'winter', s = 100, alpha = 0.5)
plt.xlabel('$x_{1}$')
plt.ylabel('$x_{1}$')
plt.grid()
```

![2020-03-01-scatter_plot](/assets/img/2020-03-01-scatter_plot.png)

데이터의 개수는 200개이다. $x$축에는 $x_{1}$ 속성, $y$축에는 $x_{2}$속성으로 독립변수를 표현하였고, 종속변수로 0’과 ‘1’의 특성을 갖는 $y$값을 각각 파란색과 초록색으로 표현하였다.

```python
X_train, X_test, y_train, y_test = train_test_split(X, y)

plt.scatter(X_train[:, 0], X_train[:, 1], c = y_train, cmap = 'winter', label = 'train data', s = 100, alpha = 0.3)
plt.scatter(X_test[:, 0], X_test[:, 1], c = 'gray', label = 'test data', s = 200, marker = '*')
plt.legend()
plt.xlabel('$x_{1}$')
plt.ylabel('$x_{2}$')
plt.grid()
```

![2020-03-01-scatter_plot_with_pred1](/assets/img/2020-03-01-scatter_plot_with_pred1.png)

sklearn의 train_test_split() 함수를 이용하여 200개의 데이터를 150개의 학습데이터와 50개의 시험데이터(회색 별)로 분류하였다.

이제 150개의 학습데이터들을 logistic_regressor 모델에 학습시켜 50개의 시험데이터(회색 별)의 특성(파란색 or 초록색)을 예측할 것이다.

```python
lr = logistic_regressor() # 위에서 만든 지도학습 알고리즘
lr.fit(X_train, y_train) # train data 적합
y_pred = lr.predict(X_test) # 예측 레이블 반환
print('예측 스코어 : ' + str(lr.score(X_test, y_test))) # 예측률 반환

plt.scatter(X_train[:, 0], X_train[:, 1], c = y_train, cmap = 'winter', label = 'train data', s = 100, alpha = 0.3)
plt.scatter(X_test[:, 0], X_test[:, 1], c = y_pred, cmap = 'winter', label = 'test data', s = 200, marker = '*')
plt.legend()
plt.xlabel('$x_{1}$')
plt.ylabel('$x_{2}$')
plt.grid()
```

![2020-03-01-scatter_plot_with_pred2](/assets/img/2020-03-01-scatter_plot_with_pred2.png)

로지스틱 회귀 모델이 절단값 0.5에서 분류를 잘 해낸것을 확인 할 수 있다. 특성값을 알 수 없던 회색 별의 색을 파란색(0) 혹은 초록색(1)으로 잘 맞추었다. 예측스코어는 1.0이 나왔고 이는 예측값과 실제값이 모두 일치해 예측이 모두 성공했음을 의미한다.

** 1.2 시그모이드 함수** 절에서 $x^{T} \beta > 0$ 이면 $1$로 예측, $x^{T} \beta < 0$ 이면 $0$으로 예측한다고 하였다. 그 말은 $x^{T} \beta = 0$는 특성값을 나누는 기준이 되며 이를 **하이퍼플레인**으로 부르게 된다. 위 모델의 하이퍼플레인 $x^{T} \beta = 0$을 시각화 하였다.

```python
print(lr.weight) # [ 5.34205998  5.75055528 -5.38943808]
weight = lr.weight

h_x1 = np.arange(-1, 5)
h_x2 = - (weight[0] + weight[1] * h_x1) / weight[2]

plt.scatter(X_train[:, 0], X_train[:, 1], c = y_train, cmap = 'winter', label = 'train data', s = 100, alpha = 0.3)
plt.scatter(X_test[:, 0], X_test[:, 1], c = y_pred, cmap = 'winter', label = 'test data', s = 200, marker = '*')
plt.plot(h_x1, h_x2, c = 'red', label = 'hyperplane')
plt.legend()
plt.xlabel('$x_{1}$')
plt.ylabel('$x_{2}$')
plt.grid()
```

![2020-03-01-scatter_plot_with_hyperplane](/assets/img/2020-03-01-scatter_plot_with_hyperplane.png)

빨간선이 하이퍼플레인이다. 2개의 변수로 나타내어지는 2차원공간에서 특성을 나누는 기준이 되기 때문에 1차원인 직선으로 나타내어졌다. 만약 3개의 변수로 나타내어지는 3차원공간에서 특성을 나누는 기준이 되었다면 2차원인 평면으로 나타내어졌을것이다. 이를 응용하여 생각하면 $k$개의 변수로 나타내어지는 $k$차원공간에서 하이퍼플레인은 $k-1$차원으로 표현되어진다는 것을 알 수 있다. 