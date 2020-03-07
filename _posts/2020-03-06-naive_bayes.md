---
layout: post
title:  "나이브 베이즈(Naive Bayes)"
subtitle:   ""
categories: data_science
tags: machine_learning
mathjax: true
comments: true
---

나이브베이즈는 **베이즈 정리**에 기초한 머신러닝 분류 알고리즘이다. 텍스트 분석, 감성 분석, 추천시스템에 강점이 있다.

# 1. 베이지안 확률론

나이브 베이즈 알고리즘은 **베이즈 정리**에 기초한다. 본 글에서는 나이브 베이즈의 이해를 위해 간단하고 짧게 설명한다. 좀 더 자세한 내용은 [이 곳]에 정리를 할 예정이다.

$B$가 조건으로 주어진 $A$의 조건부 확률 $P(A \| B)$는 다음과 같이 나타낼 수 있다.

$$
P(A | B) = \frac{P(A) P(B | A)}{P(B)}
$$

이 때, $P(A \| B)$를 사후확률(posterior), $P(A)$를 사전확률(prior), $P(B \| A)$를 우도(likelihood), $P(B)$를 관찰값(evidence)로 부르게 된다. 아래와 같이 표현할 수도 있겠다.

$$
posterior = \frac{prior \cdot likelihood}{evidence}
$$

실제 많은 상황에선 사후확률을 $P(A \| B)$를 바로 알 수 없는 경우가 많다. 이럴 경우 베이즈 정리를 이용하여, 사전확률, 우도, 관찰값을 구한 뒤, 사후확률을 구하는 경우가 많다.

# 2. 나이브 베이즈

## 2.1 확률 모델 수식

나이브 베이즈 분류기는 위의 베이즈 정리를 이용한 조건부 확률 모델이다. 독립변수가 벡터 $\bold{x} = [x_{1}, x_{2}, \cdots, x_{k}]$로 $k$개의 속성을 가지고, 종속변수가 $y$로 $i$개의 클래스를 가진다고 하자. 이 때, **독립변수의 각 속성들은 서로 독립을 가정한다.** 나이브 베이즈 분류기는 벡터 $\bold{x}$가 주어질 때, $y$가 $i$번째 클래스를 가질 확률을 다음과 같이 나타낸다.

$$
\begin{aligned}
P(C_{i} | \bold{x}) & = \frac{P(C_{i}) P(\bold{x} | C_{i})}{P(\bold{x})} \\ \\
& = \frac{P(C_{i}) P(x_{1}, x_{2}, \cdots, x_{k} | C_{i})}{P(x_{1}, x_{2}, \cdots, x_{k})}
\end{aligned}
$$

이 때, 각각의 속성들은 독립임을 가정하므로 다음과 같이 나타낼 수 있게 된다.

$$
P(C_{i} | \bold{x}) = \frac{P(C_{i}) P(x_{1} | C_{i}) P(x_{2} | C_{i}) \cdots P(x_{k} | C_{i})}{P(\bold{x})}
$$

이렇게 하여, $P(C_{i} \| \bold{x})$ 값을 최대로 하는 $i$값을 $y$의 클래스로 분류하게 되는 것이다. 이 때, $P(\bold{x})$는 $i$값에 영향을 받지 않으므로, 생략해도 상관없다.

$$
\hat{y} = arg \max_{i} \ P(C_{i}) P(x_{1} | C_{i}) P(x_{2} | C_{i}) \cdots P(x_{k} | C_{i})
$$

## 2.2 가우시안 나이브 베이즈 

속성의 분포에 따라 $P(x = v \| C_{i})$를 구하는 연산도 달라지게 된다. 분포가 가우시안 분포를 따르고 있다면, 가우시안 분포에 대한 확률 $P(x = v \| C_{i})$을, 다항 분포라면 다항 분포에 대한 확률 $P(x = v \| C_{i})$을, 베르누이 분포라면 베르누이 분포에 대한 $P(x = v \| C_{i})$을 구하여 나이브 베이즈 확률 모델에 적용 시켜야한다. 가우시안 나이브 베이즈에 사용되는 $P(x = v \| C_{i})$는 다음과 같이 구해진다.

$$
P(x = v | C_{i}) = \frac{1}{\sqrt{2 \pi \sigma_{c}^{2}}} \exp \left( -\frac{(v - \mu_{c})^{2}} {2 \sigma_{c}^{2}} \right)
$$

본 글에서는 가우시안 나이브 베이즈 모델을 만들 것이다. 만약 다항 분포나 베르누이 분포 나이브베이즈 모델을 만들고 싶다면 $P(x = v \| C_{i})$를 다항 분포나, 베르누이 분포에 대한 확률 밀도 함수로 적용시키면 될 것이다.

[위키백과](https://ko.wikipedia.org/wiki/%EB%82%98%EC%9D%B4%EB%B8%8C_%EB%B2%A0%EC%9D%B4%EC%A6%88_%EB%B6%84%EB%A5%98)로부터 예제 데이터를 가져왔다. 

**훈련 데이터**

|성별|신장 (feet)|무게 (lbs)|발의 크기 (inches)|
|:---:|:---:|:---:|:---:|
|남성|6|180|12|
|남성|5.92 (5'11")|190|11|
|남성|5.58 (5'7")|170|12|
|남성|5.92 (5'11")|165|10|
|여성|5|100|6|
|여성|5.5 (5'6")|150|8|
|여성|5.42 (5'5")|130|7|
|여성|5.75 (5'9")|150|9|

**예측 데이터**

|성별|신장 (feet)|무게 (lbs)|발의 크기(inches)|
|:---:|:---:|:---:|:---:|
|?|6|130|8|

성별(남성, 여성)을 분류하는 나이브 베이즈 모델을 만들 것이다. 이 때, 독립변수는 신장(feet), 무게(lbs), 발의 크기 (inches)가 되고, 종속변수는 성별이 된다.

각각의 속성은 **가우시안 분포**를 가정하고 있으므로, 가우시안 확률 변수의 모수에 해당하는 **평균**과 **분산**을 각 클래스 별로 구해준다.

|성별|평균(신장)|분산(신장)|평균(무게)|분산(무게)|평균(발의 크기)|분산(발의 크기)|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|남성|5.855|3.5033e-02|176.25|1.2292e+02|11.25|	9.1667e-01|
|여성|5.4175|9.7225e-02|132.5|5.5833e+02|7.5|1.6667e+00|

이를 이용하여 예측데이터 $\bold{x} = [6, 130, 8]$가 주어질 때 남성의 사후확률과 여성의 사후확률을 비교하여 더 높은 쪽을 클래스로 선택하게 된다.

- 남성의 사후확률 :

$P(\bold{x} \| male) = \frac{P(male) P(height \| male) P(weight \| male) P(footsize \| male)}{evidence}$

- 여성의 사후확률 :

\$P(\bold{x} \| female) = \frac{P(male) P(height \| female) P(weight \| female) P(footsize \| female)}{evidence}$

위에서도 말했지만 $evidence$는 대소관계 비교에서 영향을 미치지 않으므로 무시한다. 각각의 확률을 구해보도록 하자.

- $P(male)$ 은 전체 데이터의 개수 $8$개에서 각각 $4$개를 차지하므로 $\frac{4}{8}$인 $0.5$이다.
- $P(height \| male) = \frac{1}{\sqrt{2 \pi \sigma_{male}^{2}}} \exp \left( -\frac{(6 - \mu_{male})^{2}} {2 \sigma_{male}^{2}} \right)$ 이다. 이때 $\mu_{male} = 5.855$, $\sigma_{male}^{2} = 3.5033 \cdot 10^{-2}$이므로 대입시켜주면, $P(height \| male) \approx 1.5789$의 값을 얻을 수 있다.

이와 같은 방법들로 각각의 항에 대한 값을 구해주면,

- $P(weight \| male) = 5.9881 \cdot 10^{-6}$
- $P(footsize \| male) = 1.3112 \cdot 10^{-3}$

$$
\therefore P(\bold{x} | male) \approx P(male) P(height | male) P(weight | male) P(footsize | male) = 6.1984 \cdot 10^{-9}
$$

- $P(female) = 0.5$
- $P(height \| female) = 2.2346 \cdot 10^{-1}$
- $P(weight \| female) = 1.6789 \cdot 10^{-2}$
- $P(footsize \| female) = 2.8669 \cdot 10^{-1}$

$$
\therefore P(\bold{x} | female) \approx P(female) P(height | female) P(weight | female) P(footsize | female) = 5.3778 \cdot 10^{-4}
$$

여성의 사후확률이 남성의 사후확률보다 크기 때문에 여성으로 예측한다.

# 3. 파이썬 코드

가우시안 나이브베이즈 분류 모델을 파이썬 코드로 구현해 보았다. fit 속성으로 훈련데이터를 적합시키고, predict 속성으로 특성을 예측하게 된다. score 속성으로 정확도를 나타내도록 하였다.

```python
import numpy as np

class naive_bayes_classifier:
    print('가우시안 정규분포를 가정한 나이브베이즈 모형입니다.')
        
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        
        y_label_list, y_label_counts = np.unique(y_train, return_counts = True)
        self.y_label_list = y_label_list
        self.y_label_counts = y_label_counts
        
        parameter = []
        
        for label in y_label_list:
            label_boolean = (self.y_train == label).flatten()
            mean_var_list = []
            
            for v in range(self.X_train.shape[1]):
                mean = np.mean(X_train[label_boolean, v])
                var = np.var(X_train[label_boolean, v])
                mean_var_list.append({'mean' : mean,
                                      'var' : var})
        
            parameter.append(mean_var_list)
            
        self.parameter = parameter
        return self.parameter
        
    def predict(self, X_test):
        
        self.X_test = X_test
        
        y_pred = []
        
        for x in range(len(self.X_test)):
            posterior_list = []
            
            for label_loc in range(len(self.y_label_list)):
                prior = self.y_label_counts[label_loc] / len(self.y_train)
                likelihood = 1
                
                for v in range(self.X_train.shape[1]):
                    mean = self.parameter[label_loc][v]['mean']
                    var = self.parameter[label_loc][v]['var']
                    
                    likelihood *= np.exp(-((self.X_test[x, v] - mean) ** 2) / (2 * var)) / (np.sqrt(2 * var))
                
                posterior_list.append(prior * likelihood)
            
            max_loc = np.argmax(posterior_list)
            y_pred.append(self.y_label_list[max_loc])
        
        return np.array(y_pred)
    
    def score(self, X_test, y_test):
        
        return np.mean(self.predict(X_test) == y_test)
```

가우시안 나이브 베이즈 모델의 예측과정을 시각적으로 나타내기 위해 sklearn 모듈의 make_blobs 데이터셋을 가져왔다.

```python
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
from sklearn.model_selection import train_test_split

X, y = make_blobs(n_samples = 200, centers = 2, n_features=2, random_state=0, cluster_std = 0.4)

plt.scatter(X[:, 0], X[:, 1], c = y, cmap = 'winter', s = 100, alpha = 0.5)
plt.xlabel('$x_{1}$')
plt.ylabel('$x_{2}$')
plt.grid()
```

![2020-03-06-scatter_plot](/assets/img/2020-03-06-scatter_plot.png)

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

![2020-03-06-scatter_plot_with_pred1](/assets/img/2020-03-06-scatter_plot_with_pred1.png)

sklearn의 train_test_split() 함수를 이용하여 200개의 데이터를 150개의 학습데이터와 50개의 시험데이터(회색 별)로 분류하였다.

이제 150개의 학습데이터들을 naive_bayes_regressor 모델에 학습시켜 50개의 시험데이터(회색 별)의 특성(파란색 or 초록색)을 예측할 것이다.

```python
nb = naive_bayes_classifier() # 위에서 만든 지도학습 알고리즘
nb.fit(X_train, y_train) # train data 적합
y_pred = nb.predict(X_test) # 예측 레이블 반환
print('예측 스코어 : ' + str(nb.score(X_test, y_test))) # 예측률 반환

plt.scatter(X_train[:, 0], X_train[:, 1], c = y_train, cmap = 'winter', label = 'train data', s = 100, alpha = 0.3)
plt.scatter(X_test[:, 0], X_test[:, 1], c = y_pred, cmap = 'winter', label = 'test data', s = 200, marker = '*')
plt.legend()
plt.xlabel('$x_{1}$')
plt.ylabel('$x_{2}$')
plt.grid()
```

![2020-03-06-scatter_plot_with_pred2](/assets/img/2020-03-06-scatter_plot_with_pred2.png)

가우시안 나이브 베이즈 모델이 분류를 잘 해낸것을 확인 할 수 있다. 특성값을 알 수 없던 회색 별의 색을 파란색(0) 혹은 초록색(1)으로 잘 맞추었다. 예측스코어는 1.0이 나왔고 이는 예측값과 실제값이 모두 일치해 예측이 모두 성공했음을 의미한다.