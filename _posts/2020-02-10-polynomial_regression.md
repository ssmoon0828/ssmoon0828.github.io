---
layout: post
title:  "다항회귀와 릿지, 라쏘, 엘라스틱넷"
subtitle:   ""
categories: data_science
tags: machine_learning
mathjax: true
comments: true
---

머신러닝을 시행할땐 **과적합(Overfitting)**을 유념해야한다. 다항회귀(Polynomial Regression)는 선형회귀에 비해 과적합이 일어나기 쉽다. 과적합을 막을 수 있는 릿지(Ridge), 라쏘(Lasso), 엘라스틱넷(ElasticNet)에 대해 포스팅한다.

# 1. 다항회귀(Polynomial Regression)

독립변수과 종속변수과의 관계를 언제나 일차식으로 모델링하기엔 무리가 있다. 아래의 이미지를 보자.

![2020-02-10-scatter_plot_2](/assets/img/2020-02-10-scatter_plot_2.png)

위와 같은 데이터셋에서 데이터들의 추세를 [선형회귀](https://ssmoon0828.github.io/data_science/2020/02/05/linear_regression/)를 통하여 일직선으로 표현하기엔 어색하다. 이차함수의 곡선이 더 적합해 보인다.

![2020-02-10-scatter_plot_2_with_polynomial](/assets/img/2020-02-10-scatter_plot_2_with_polynomial.png)

선형회귀(노란선)와 다항회귀(빨간선)를 적용한 결과이다. 선형회귀선(노란선)은 오차를 가장 적게 설명하는 일직선을 표현했음에도 데이터의 경향을 충분히 반영하지 못한 것으로 보인다. 그에 비해 오차를 가장 적게 설명하는 이차식을 표현한 다항회귀선(빨간선)은 데이터의 경향을 잘 나타낸 것으로 보인다.

이처럼 일차함수식보단 이차 이상의 함수식으로 독립변수와 종속변수의 상관관계를 모델링 하는 것을 **다항회귀(Polynomial Regression)**라 한다.

## 1-1. 아이디어 & 수식

다항회귀식을 만드는 방법은 간단하다. 먼저 독립변수와 종속변수와의 상관관계를 몇 차식으로 표현할 것인지 정한다. $k$차식으로 표현하기로 정했다면 독립변수인 $x$에 대한 벡터를 각각의 원소에 차수를 하나씩 증가시켜 다음과 같은 행렬 $X$로 표현한다.

$$
x = \begin{bmatrix}
x_{1} \\
x_{2} \\
\vdots \\
x_{n}
\end{bmatrix}
\rightarrow
X = \begin{bmatrix}
1 & x_{1} & (x_{1})^2 & \cdots & (x_{1})^k \\
1 & x_{2} & (x_{2})^2 & \cdots & (x_{2})^k \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
1 & x_{n} & (x_{n})^2 & \cdots & (x_{n})^k
\end{bmatrix}
$$

독립변수($X$)의 열벡터의 개수에 따라 회귀계수의 개수 또한 $k+1$개로 늘어났으므로, 회귀 계수에 대한 벡터($\beta$)도 $k+1$개의 원소를 갖는다.

$$
\beta = \begin{bmatrix}
\beta_{0} \\
\beta_{1} \\
\vdots \\
\beta_{k}
\end{bmatrix}
$$

이제 종속변수와 독립변수와의 관계는 $y = X\beta + \epsilon$ 으로 표현할 수 있기 때문에 오차제곱합(RSS)를 가장 작게 만드는 회귀계수벡터($\beta$)를 아래와 같이 추정하면 다항회귀 모델이 완성된다. 이 부분이 이해가 안간다면 [선형회귀](https://ssmoon0828.github.io/data_science/2020/02/05/linear_regression/) 포스팅을 보고 오는 것을 추천한다.

$$
\beta = (X^T X)^{-1} (X^T y)
$$

## 1-2. 주의점

하지만 다항회귀를 통한 모델링은 데이터가 충분하지 않거나 차수가 높아지면 **과적합(Overfitting)**이 될 우려가 있다. 

![2020-02-10-warning](/assets/img/2020-02-10-warning.png)
*이미지 출처 : https://brunch.co.kr/@itschloe1/11*

때문에 릿지(Ridge), 라쏘(Lasso), 엘라스틱넷(ElasticNet)과 같은 규제를 걸어주어 오버피팅을 완화시켜주는 작업이 필요하다.

# 2. 릿지(Ridge)

## 2.1. 아이디어

릿지회귀는 오차제곱합($\sum_{i}^{n} (\epsilon_{i})^{2}$)에 페널티항($\lambda \sum_{i}^{k} (\beta_{i})^{2}$) 항을 추가시켜 회귀계수벡터($\beta$)를 추정한다. 여기서 $\lambda$는 스칼라이다. 수식으로 나타내면 다음과 같다.

$$
\beta_{ridge} = arg\min_{\beta}(X^T X)^{-1} (X^T y) + \lambda \sum_{i}^{k} (\beta_{i})^2
$$

여기서 $\lambda$값을 이용하여 규제의 정도를 조절할 수 있다. $\lambda$값을 0으로 설정하면 일반 다항회귀의 회귀계수 추정식과 같게된다. 릿지회귀를 통해 추정된 회귀계수는 다음과 같이 증명할 수 있다.

$$
\begin{aligned}
RSS_{ridge} & = (y - X\beta)^T (y - X\beta) + \lambda\beta^T \beta \\
& = y^T y - y^T X\beta - \beta^T X^T y + \beta^T X^T X \beta + \lambda \beta^T \beta
\end{aligned}
$$

$$
\begin{aligned}
\frac{\partial RSS_{ridge}}{\partial \beta} = -2 X^Ty + 2 X^T X \beta + 2 \lambda\beta = 0 \\\\
\therefore \beta = (X^T X + \lambda I)X^T y
\end{aligned}
$$

## 2.2. 파이썬 코드

다항회귀와 릿지회귀를 polynomial_regressor라는 이름의 클래스로 구현해 보았다. 기본적으로 다항회귀지만 alpha 매개변수를 통해 릿지회귀도 같이 구현할 수 있다. *(python에서 lambda는 무명함수를 생성할 때 쓰는 명령어기 때문에 alpha로 대체하였다.)*

```python
import numpy as np

class polynomial_regressor:
        
    def __init__(self, bias = True, degree = 2, alpha = 0):
        '''
        편향의 유무, 다항식의 차수, 릿지를 설정할 수 있다.
        '''
        self.bias = bias
        self.degree = degree
        self.alpha = alpha
    
    def fit(self, X_train, y_train):
        '''
        회귀계수를 구할 수 있다.
        X_train은 1차원이어야 한다.
        '''
        self.X_train = X_train.reshape(len(X_train), 1)
        
        polynomial_list = []
        
        for i in range(2, self.degree + 1):
            polynomial_list.append(self.X_train ** i)
        
        X_train_add = np.hstack(polynomial_list)
        self.X_train = np.hstack([self.X_train, X_train_add])
        
        if self.bias == True:
            one_vector = np.ones([len(self.X_train), 1])
            self.X_train = np.hstack([one_vector, self.X_train])
        
        self.y_train = y_train.reshape(len(y_train), 1)
        
        mat1 = np.linalg.inv(np.dot(self.X_train.T, self.X_train) + self.alpha * np.eye(len(self.X_train[0])))
        mat2 = np.dot(self.X_train.T, self.y_train)
        weight = np.dot(mat1, mat2)
        self.weight = weight
        
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

코드를 테스트 하기 위해 데이터셋을 임의로 만들어 보았다. $y = -10 + 24x -9x^2 +  x^3$의 다항식에 독립변수와 종속변수에 오차인 랜덤변수를 추가시켜 데이터를 흩트려 놓았다. $y^\prime = 3(x-2)(x-4)$이므로 $x=2$, $x=4$에서 변곡점을 갖는다.

```python
np.random.seed(0)

x = np.random.rand(100) * 5 + 0.5
y = x**3 -9 * x**2 + 24 * x - 10 + np.random.rand(100) * 5
data = np.hstack([x.reshape(len(x), 1), y.reshape(len(y), 1)])

scatter_plot = plt.figure(figsize = (8, 6))
plt.scatter(data[:, 0], data[:, 1], label = 'data point')
plt.xlabel('x')
plt.ylabel('y')
plt.grid()
plt.legend()
scatter_plot.savefig('scatter_plot.png', dpi = 200)
```

![2020-02-10-scatter_plot](/assets/img/2020-02-10-scatter_plot.png)

위와 같이 만들어진 데이터 셋에 polynomial_regressor클래스로 다항회귀식의 회귀계수들을 구했다. 릿지회귀는 사용하지 않기 때문에 alpha는 디폴트 값인 0이다.

```python
pr = polynomial_regressor(degree = 3)
pr.fit(data[:, 0], data[:, 1])
weight = pr.weight 
print(weight) # weight : [-7.33519131 23.726838   -8.77123439  0.96164087]

def polynomial3(weight, x):
    
    return weight[0] + weight[1] * x + weight[2] * x**2 + weight[3] * x**3

x = np.arange(0.5, 5.5, 0.1)
y = polynomial3(weight, x)

scatter_plot_with_regressor_line = plt.figure(figsize = (8, 6))
plt.scatter(data[:, 0], data[:, 1], label = 'data point')
plt.plot(x, y, color = 'red', label = 'regressor line')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.grid()
plt.legend()
scatter_plot_with_regressor_line.savefig('scatter_plot_with_regressor_line.png', dpi = 200)
```

회귀계수 벡터는 [-7.33519131 23.726838   -8.77123439  0.96164087]가 나왔다. 설정했던 [-10, +24, -9, 1]과 비슷하게 나왔고 랜덤변수로 오차를 생성시켰기 때문에 정확하게 일치하진 않았다. 이에 따라 다항 회귀식을 이용한 추세선을 그린 결과 다음과 같은 plot을 얻을 수 있었다.

![2020-02-10-scatter_plot_with_regressor_line](/assets/img/2020-02-10-scatter_plot_with_regressor_line.png)

이번엔 이렇게 구해진 다항회귀에 릿지회귀를 추가하였다. alpha값을 1, 0.1, 10으로 변화시키면서 확인한 결과 다음과 같은 plot을 얻을 수 있었다.

```python
pr1 = polynomial_regressor(degree = 3, alpha = 1)
pr1.fit(data[:, 0], data[:, 1])
pr1_weight = pr1.weight
y_pr1 = polynomial3(pr1_weight, x)

pr01 = polynomial_regressor(degree = 3, alpha = 0.1)
pr01.fit(data[:, 0], data[:, 1])
pr01_weight = pr01.weight
y_pr01 = polynomial3(pr01_weight, x)

pr10 = polynomial_regressor(degree = 3, alpha = 10)
pr10.fit(data[:, 0], data[:, 1])
pr10_weight = pr10.weight
y_pr10 = polynomial3(pr10_weight, x)


scatter_plot_with_ridge = plt.figure(figsize = (8, 6))
plt.scatter(data[:, 0], data[:, 1], label = 'data point')
plt.plot(x, y, color = 'red', label = 'regressor line')
plt.plot(x, y_pr1, color = 'green', label = 'alpha : 1')
plt.plot(x, y_pr01, color = 'orange', label = 'alpha : 0.1')
plt.plot(x, y_pr10, color = 'purple', label = 'alpha : 10')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.grid()
plt.legend()
scatter_plot_with_ridge.savefig('scatter_plot_with_ridge.png', dpi = 200)
```

![2020-02-10-scatter_plot_with_ridge](/assets/img/2020-02-10-scatter_plot_with_ridge.png)

페널티항의 $\lambda$값이 0에 가까워질수록 원래 다항회귀선과 가까워지지만 $\lambda$값이 커지면서 선의 굴곡이 느슨해지는 것을 확인할 수 있다. 또한 이것은 **과적합(Overfitting)**을 막는 과정으로 해석 할 수 있다.

만약 데이터셋의 크기가 충분하지 않거나 차수가 매우 높았다면 모델이 학습데이터에만 과적합 되기 때문에, 굴곡을 느슨하게 하면서 분산을 낮추는 과정이 필요했을 것이다. 릿지 회귀를 이용하면 $\lambda$값을 조절하면서 오버피팅현상을 피할 수 있게된다.

# 3. 라쏘(Lasso), 엘라스틱넷(ElasticNet)

라쏘와 엘라스틱넷 또한 릿지와 마친가지로 오차제곱합(RSS)에 페널티항을 추가시켜 회귀계수를 추정한다. 수식은 다음과 같다.

$$
\begin{aligned}
\beta_{lasso} = arg\min_{\beta}(X^T X)^{-1} (X^T y) + \lambda \sum_{i}^{k} \left\vert \beta_{i} \right\vert \\
\beta_{elastic} = arg\min_{\beta}(X^T X)^{-1} (X^T y) + \lambda_{1} \sum_{i}^{k} (\beta_{i})^2 + \lambda_{2} \sum_{i}^{k} \left\vert \beta_{i} \right\vert
\end{aligned}
$$

릿지는 회귀계수들의 제곱합을 페널티항으로 추가했지만 라쏘는 회귀계수들의 절댓값들의 합을 페널티항으로 추가한다. 엘라스틱넷은 릿지의 페널티항과 라쏘의 페널티항을 모두 추가하였다.

릿지든 라쏘든 엘라스틱넷이든 오버피팅을 피하기 위해선 적절한 페널티항의 계수 $\lambda$값을 찾는 것이 중요할 것이다. 
