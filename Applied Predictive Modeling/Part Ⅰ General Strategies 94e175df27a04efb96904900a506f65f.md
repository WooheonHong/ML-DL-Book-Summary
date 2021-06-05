# Part Ⅰ General Strategies

# Ch2. A short Tour of the Predictive Modeling Process

## Case Study: Predicting Fuel Economy

- 선형회귀 적합

![Part%20%E2%85%A0%20General%20Strategies%2094e175df27a04efb96904900a506f65f/Untitled.png](Part%20%E2%85%A0%20General%20Strategies%2094e175df27a04efb96904900a506f65f/Untitled.png)

![Part%20%E2%85%A0%20General%20Strategies%2094e175df27a04efb96904900a506f65f/Untitled%201.png](Part%20%E2%85%A0%20General%20Strategies%2094e175df27a04efb96904900a506f65f/Untitled%201.png)

- 이차항 추가

$$efficiency = 63.2 - 11.9 × displacement + 0.94 × displacement^2$$

![Part%20%E2%85%A0%20General%20Strategies%2094e175df27a04efb96904900a506f65f/Untitled%202.png](Part%20%E2%85%A0%20General%20Strategies%2094e175df27a04efb96904900a506f65f/Untitled%202.png)

quadratic model의 한 가지 문제점은 예측변수의 끝 값에서 잘 작동하지 못한다는 점이다. Fig 2.3을 보면 배기량이 높은 차량에 대해서는 잘 예측하지 못할 수 있다. 

이에 대한 대안으로 multivariate adaptive regression spline(MARS)

- MARS

![Part%20%E2%85%A0%20General%20Strategies%2094e175df27a04efb96904900a506f65f/Untitled%203.png](Part%20%E2%85%A0%20General%20Strategies%2094e175df27a04efb96904900a506f65f/Untitled%203.png)

결과적으로 RMSE는 quadratic model과 거의 동일하지만 끝값에서 잘 작동하지 않는 단점을 고려했을 때 MARS를 선택하자. 

# Ch3. Data Pre-processing

데이터 전처리는 사용되고 있는 모델에 따라 달라진다. 예를 들어, tree-based model은 예측 변수의 특징에 덜 민감하지만 선형 회귀는 민감하다.

## Case Study: Cell Segmentation in High-Content Screening

약의 효과나 살아 있는 유기체에서 세포의 크기, 모양, 발당 상태 그리고 수를 이해하는 것에 관심이 있다. 이런 샘플을 측정하는 방법으로 high-content screening을 사용한다. 

- 총 2019개의 세포 중 1300개는 색 경계로 잘 분리되지 못했고 719개는 잘 분리됐다.
- 1009개의 세포가 훈련셋에 존재한다.
- 116개의 피쳐가 측정되었고 세포 segmentation 품질을 예측하기 위해 사용된다.

## Data Transformations for Individual Predictors

- Centering and Scaling

유일한 단점은 개별 값에 대한 해석 능력 손실

- Transformations to Resolve Skewness

관습적으로 max / min > 20 이면 상당한 skewness를 가진다. 또 다른 방법으로는 skewness 통계량이 0에서 멀어질수록 right(left) skewness를 가진다. 

 

### Box-Cox Transformation

$$x^{*} = \begin{dcases} \frac{x^\lambda - 1}{\lambda} &\text{if } ~\lambda =\not 0   \\ log(x) & \text{if } ~ \lambda = 0 \end{dcases}$$

69개의 변수는 0과 음수 때문에 변환이 되지 않고 3개는 $\lambda = +- 0.02$ 내에 있으므로 변환하지 않는다.

## Data Transformation for Multiple Predictors

### Transformations to Resolve Outliers

- 이상치가 의심될 때 첫 번째 단계는 값이 과학적으로 타당한지 확인
- 데이터 수집 방법에 따라 이상치의 정의가 바뀐다.
- 이상치에 덜 민감한 model: Tree-based 분류 모델, SVM 분류 모델
- 모델이 이상치에 민감하다면 spatial sign을 사용해볼 수 있다.

    $$x_{ij}^* = \frac{x_{ij}}{\Sigma_{j = 1}^{p}x_{ij}^2}$$

    예측 변수의 값을 다차원 구로 사영시킨다. 모든 샘플을 구로부터 같은 거리에 있기 한다. 주의할 점은 예측변수들을 group으로 생각하므로 변수를 제거하는 것은 문제를 일으킨다.

### Data Reduction and Feature Extraction

**PCA**

- 몇 가지 예측모델은 해를 찾기 위해 그리고 모델의 수치적 안정성을 개선하기 위해 상관되지 않은 예측 변수를 선호한다.
- PCA는 측정 단위와 분포 같은 예측 변수에 대한 이해가 없으며 모델링 목적에 대한 이해가 없이 진행된다.  단순히 예측 변수 집합의 변동성을 추구한다는 것을 반드시 알아야 한다.
- 먼저 skewed 예측 변수를 변환하고 그리고 나서 표준화하자.
- PCA의 supervised learning 버전이 PLS이다.
- 성분의 수를 결정하기 위해 **scree plot**을 그려보자.

    만약 PCA가 데이터의 충분한 정보를 담고 있다면 처음 몇 가지 성분에 대해 plot을 그려보자. 이러한 플랏은 샘플의 클러스터나 개별 데이터의 면밀한 검사를 유발하는 이상치를 설명할 수 있다. 분류 문제에서 PCA plot은 잠재적 클래스 분류를 보일 수도 있다. 이것은 모델러의 최초 기대를 정할 수 있다. 플랏을 그릴 때 유의할 점은 작은 분산을 설명하는 성분일수록 scale이 작아지는 경향이 있다는 것이다.

- loading을 통해 어떤 예측 변수가 주성분에 가장 큰 영향을 끼치는지 알 수 있다. loading이 0에 가까울수록 해당 주성분에 대한 기여가 낮다. 하지만 데이터의 분산을 잘 설명하더라도 반응 변수 예측에 기여가 높은 것은 아니다.

## Dealing with Missing Values

### **informative missingness**

결측의 패턴이 outcome과 관련되어 있는지 확인하자.

informative missingness는 모델에 상당한 편향을 포함할 수 있다. 예를 들어, 사람들은 강한 의견을 가질 때 상품을 평가하는 경향이 있다 .따라서 데이터는 평가 항목의 중간이 거의 없는 상태로 양극화될 가능성이 높다. 

### **Censored data**

censored data는 데이터의 측정값이나 관찰치가 부분적으로만 알려진 상태를 의미한다.

censored data와 missing data는 동일하지 않다에 주의하자. 

해석이나 추론에 초점을 맞춘 전통적은 통계 모델을 세울 때 중도 절단은 censoring 메커니즘에 관한 가정을 세움으로써 보통 설명된다. 예측 모델에서는 censored data를 단순히 결측치로 다루거나 관측치로써 다루는 경우가 많다. 예를 들어, 표본이 관측의 한계를 넘지 않는 값을 가질 때 실제 한계는 실제 값 대신에 사용될 수 있다. 이런 상황에서는 0과 관측 한계 사이의 난수를 생성하는 것도 흔한 방법이다. 

### Missing data

보통 결측치는 모든 예측 변수에서 임의로 나타나는 것이 아닌 예측 변수의 일부에 집중되어 있다.

결측치가 특정 샘플에 집중되어 있을 수도 있다. 큰 데이터에서는 결측치가 유의하지 않다면 제거해도 되지만 작은 데이터에서는 몇 가지 대안을 사용하는 것이 적절하다.

1. 결측치 제거
2. tree-based 모델과 같은 몇 가지 예측 모델은 결측치를 설명할 수 있다.
3. 훈련 셋의 예측 변수 정보를 이용하여 결측치 대치 

    결측치 대치는 불확실성을 추가한다. 모델의 성능을 추정하거나 파라미터 튜닝을 선택하기 위해 리샘플링을 사용한다면, 결측치 대치는 resampling 내에서 통합되어야만 한다. 

    결측치에 영향을 받는 예측변수의 수가 적다면, 예측 변수간의 관계에 대한 탐색 방법이 좋은 방법이다. PCA같은 시각화로 예측 변수 간 상관관계가 있는지 확인하자. 결측치가 있는 변수가 결측치가 거의 없는 다른 변수와 강하게 상관관계가 있다면 대치하기에 좋다. 

    대치에 인기있는 방법은 KNN이다. 훈련셋의 가장 가까운 샘플들을 찾아서 평균내어 결측치를 대치한다. 이 방법의 장점은 훈련셋 범위 내에서 대치된 데이터가 정제된다는 것이다. 단점은 전체 훈련셋에 대하여 결측값 대치가 필요할 때 매 번 필요하다는 것이다. nearest neighbor 방법은 결측치의 수 뿐 아니라 neighbor의 수에도 robust하다.  

## Removing Predictors

모델링 전에 예측변수를 삭제하는 이점으로는

1. 계산 시간과 복잡도 감소
2. 상관관계가 높은 변수를 삭제하는 것이 모델 성능과 타협할 수는 없지만 parsimonious(모수의 수가 적어서 간단하지만 데이터의 구조를 잘 설명하는)하고 해석 가능한 모델을 만든다. 
3. 몇 가지 모델은 degenerate 분포를 가진 예측변수 때문에 작동이 안될 수 있다. 이런 문제 있는 변수들을 제거하면 모델의 성능과 안정성에 큰 개선이 이뤄질 수 있다. 예측 변수가 단 하나의 값 만을 가지는 경우(즉, 분산이 0인 예측변수), 몇 가지 모델에 대해서는 계산에 거의 영향을 미치지 않는다. 예를 들면, tree-based 모델에서는 split에 사용되지 않기 때문에 이런 타입의 변수는 영향을 미치지 않는다. 반면에 선형회귀와 같은 모델에서는 계산 에러가 발생한다.

    마찬가지로 몇 가지 예측 변수는 오직 매우 적은 빈도의 소수의 유일한 값만을 가질 수 있다. 

    **near-zero 분산을 가지는 예측 변수를 정하는 경험적 규칙** 

    - 표본 크기에 대한 유일한 값의 비율이 10% 이하
    - 두 번째로 빈도수가 많은 값에 대한 첫 번째로 빈도수가 많은 값의 비율이 20 이상

caret에서 `preProcess = "zv"`를 설정해주면 분산이 없거나, 매우 낮은 분산을 가지는 변수를 제거해준다. 

### Between-Predictor corrleations

Correlation plot에서 변수의 정렬을 클러스터에 기반하여 정하면 보기에 좋다.

선형 모델에서는 VIF를 사용하면 다중공선성을 확인할 수 있다. 하지만 문제 해결을 위해  어떤 변수를 제거해야 할 지에 대해서는 알려주지 않는다. 대신에 덜 이론적이지만 좀 더 휴리스틱한 접근법이 있다. 모든 쌍의 상관계수의 값이 특정 경계선 아래에 있도록 최소한의 예측 변수를 제거하는 것이다. 오직 2차원의 다중공선성만 확인하지만 몇 가지 모델의 성능에 상당히 긍정적인 영향을 준다. 

알고리즘

1. 예측변수의 상관계수 행렬 계산
2. 상관계수가 가장 높은 두 예측변수 A, B를 결정
3.  A와 다른 변수들의 상관계수의 평균을 구하고 B와 다른 변수들의 상관계수의 평균을 구한다. 
4. 이 중에서 높은 값을 가지는 변수에 대해 제거한다.
5. 모든 상관계수 값이 threshold를 넘지 않도록 2-4 단계를 계속해서 반복한다. 

주의할 점은 outcome은 고려하지 않는 방법이라는 것이다.

## Adding Predictors

모든 더미 변수를 넣을 때, itercept항이 포함된 모델은 수치적 문제가 있을 수 있으므로 주의 

간단한 모델들은 사용자가 예측 변수의 비선형 항을 추가해줘야 할 수도 있다. x + x^2 + ...

## Binning Predictors

bin 장점

1. 간단한 결정 규칙, 간단한 해석
2. 모델러가 예측 변수와 outcome의 정화갛ㄴ 관계를 알 필요 x
3. 높은 설문조사 응답률

bin 단점

1. 모델 성능 손실
2. 단순한 예측
3. 높은 false positive 

## Exercises

1. The UC Irvine Machine Learning Repository1 contains a data set related to glass identication. The data consist of 214 glass samples labeled as one of seven class categories. There are 9 predictors, including the refractive index and percentages of 8 elements: Na, Mg, Al, Si, K, Ca, Ba and Fe. The data can be accessed via:

(a) Using visualizations, explore the predictor variables to understand their distributions as
well as the relationships between predictors.

![Part%20%E2%85%A0%20General%20Strategies%2094e175df27a04efb96904900a506f65f/Untitled%204.png](Part%20%E2%85%A0%20General%20Strategies%2094e175df27a04efb96904900a506f65f/Untitled%204.png)

Figure 1: Density plots of each of the predictors in the original Glass data set. The points along the x-axis show the values of the individual samples

K 와 Mg에서 두 번째로 큰 최빈값이 0주위에서 존재한다. Ca, Ba, Fe, RI는 약간의 skewness를 보인다.   K에서 한 두개의 이상치가 있을 수 있지만, 단순히 자연스러운 skewness 때문일 수도 있다. 또한, Ca, RI, Na, 그리고 Si가 분포의 가장자리에서 적은 수의 측정값을 가지고 scale의 중간에서 샘플이 집중된다. 이러한 특징은 두터운 꼬리를 가지는 분포를 나타낸다. 

(b) Does there appear to be any outliers in the data? Are predictors skewed?

K의 이상치로 의심되는 점들은 단순히 샘플의 수가 충분하지 않아 나타나는 것일 수도 있기 때문에 모델링을 통해서 설명되어야 한다. 이상치에 강건한 모델을 사용하는 것이 선호될 수 있다. 

(c) Are there any relevant transformations of one or more predictors that might improve the
classication model?

Box-Cox 변환은 0을 변환시키지 못하므로 대신에 Yeo-Johnson 변환을 사용한다. 

---

**Yeo-Johnson Transformation**

![Part%20%E2%85%A0%20General%20Strategies%2094e175df27a04efb96904900a506f65f/Untitled%205.png](Part%20%E2%85%A0%20General%20Strategies%2094e175df27a04efb96904900a506f65f/Untitled%205.png)

양수일때 y+1의 box-cox 변환이고 음수일때 -y+1의 box-cox 변환

---

![Part%20%E2%85%A0%20General%20Strategies%2094e175df27a04efb96904900a506f65f/Untitled%206.png](Part%20%E2%85%A0%20General%20Strategies%2094e175df27a04efb96904900a506f65f/Untitled%206.png)

Fig 3. Density plots of the Glass predictors after a Yeo-Johnson transofrmation

두 번째 최빈값이 Ba와 Fe에서 보인다. 

`preProcess(Glass[, -10], method = "YeoJohnson")` 가 skewness를 개선하지 못했으므로 **spatial sign** 변환을 사용해 이상치를 완화시켰다.  ****

![Part%20%E2%85%A0%20General%20Strategies%2094e175df27a04efb96904900a506f65f/Untitled%207.png](Part%20%E2%85%A0%20General%20Strategies%2094e175df27a04efb96904900a506f65f/Untitled%207.png)

Fig4. A scatterplot matrix of the Glass data after the spatial sign transformation

가능한 많은 이상치들이 데이터의 주류로 수축되었다. 이 변환은 적어도 하나의 새로운 패턴을 초래했다. Fe와 B에 대한 0 값을 가진 샘플들이 이차원의 직선으로 투영되었다. skewness를 해결할 수는 없었지만, 극단치를 최소화 할 수 있었다. 예측 변수 분포 문제를 해결하기 위한 전처리에 대한 시도가 항상 성공하지는 않는다. 전처리의 최적의 노력은 매우 바람직한 변환된 값을 산출하지 않을 지도 모른다. 이러한 종류의 상황에서 skewed 분포에 의해 지나치게 영향을 받지 않는 모델들을(tree-based 모델등) 사용할 필요가 있다. 

2. The Soybean data can also be found at the UC Irvine Machine Learning Repository. Data were
collected to predict disease in 683 soybeans. The 35 predictors are mostly categorical and include
information on the environmental conditions (e.g. temperature, precipitation) and plant conditions
(e.g. left spots, mold growth). The outcome labels consist of 19 distinct classes

(a) Investigate the frequency distributions for the categorical predictors. Are the distributions
likely to cause issues for models.

우선 temp 변수의 정보가 유익하지 않으므로  `$ temp : Ord.factor w/ 3 levels "0"<"1"<"2": 2 2 2 2 2 2 2 2 2 2 ...`  아래와 같이 변환 

```r
library(car)
> Soybean2$temp <- recode(Soybean2$temp,
+ "0 = 'low'; 1 = 'norm'; 2 = 'high'; NA = 'missing'",
+ levels = c("low", "norm", "high", "missing"))
> table(Soybean2$temp)
```

 month와 강수량도 변환.

두 범주형 변수에 대한 joint 분포는 분할표 사용 + 적절한 시각화 

(b) Roughly 18% of the data are missing. Are there particular predictors that are more likely
to be missing? Is the pattern of missing data related to the classes?

(c) Develop a strategy for dealing with the missing data, either by eliminating predictors or
imputation

결측치를 다루기 위해 대치법을 사용할 수 있다. 하지만, 예측 변수의 거의 100%가 몇 가지 경우에 대하여 대치될 필요가 있기 때문에 대치법은 도움이 되지 않을 수 있다.  결측치를 다른 수준으로 인코딩 하거나 높은 비율을 가지는 결측치와 연관된 클래스를 제거할 수 있다. 

예측변수 값의 빈도가 모델링 과정에 어떻게 영향을 끼칠까? 만약 sparsity에 민감한 모델을 사용한다면, 낮은 비율의 팩터 레벨은 문제가 될 수 있다.  팩터를 더비변수로 변환해서 sparsity가 좋은지 나쁜지를 확인할 수 있다. 

near-zero 분산을 확인해서 어떤 변수를 제거할지 결정하자. 다른 방법은 tree 모델 같은 rule-based 모델을 사용하거나 나이브 베이즈 모델을 사용하자. 

3.  Chapter 5 introduces Quantitative Structure{Activity Relationship (QSAR) modeling where the
characteristics of a chemical compound are used to predict other chemical properties. The caret
package contains such a data set from (Mente & Lombardo 2005). Here, where the ability of a
chemical to permeate the blood{brain barrier was experimentally determined for 208 compounds.
134 predictors were measured for each compound.

(b) Do any of the individual predictors have degenerate distributions?

예측변수들의 skewness에 대한 box plot을 그려보자. yeo-johnson 변환을 사용한 결과를 보자.

![Part%20%E2%85%A0%20General%20Strategies%2094e175df27a04efb96904900a506f65f/Untitled%208.png](Part%20%E2%85%A0%20General%20Strategies%2094e175df27a04efb96904900a506f65f/Untitled%208.png)

대칭이 되거나 추가적으로 최빈값이 나타나게 된다. 

(c) Generally speaking, are there strong relationships between the predictor data? If so, how
could correlations in the predictor set be reduced? Does this have a dramatic eect on the
number of predictors available for modeling?

skewed 되어 있는 데이터가 많은 것이 문제가 된다. 상관계수는 예측 변수를 제곱한 값의 함수이기 때문에 꼬리는 상관계수에 상당한 영향을 준다. 이러한 이유로 상관계수 구조를 세 가지 방법으로 볼 것이다. 변환하지 않은 데이터, Yeo-Johnson 변환 이후의 데이터, spatial sign 변환 이후의 데이터

![Part%20%E2%85%A0%20General%20Strategies%2094e175df27a04efb96904900a506f65f/Untitled%209.png](Part%20%E2%85%A0%20General%20Strategies%2094e175df27a04efb96904900a506f65f/Untitled%209.png)

Fig7. Correlation matrices for the raw data(top), transformed via the Yeo-Johnson transformation(middle) and the spatial sign transformation(bottom)

예측변수간 상관관계를 풀기 위해 데이터를 변환시키기보다, 예측변수를 제거하자. 

# Ch4. Over-Fitting and Model Tuning

이 챕터에서는 데이터 품질이 충분하고 전체 모집단을 대표한다고 가정

## The Problem of Over-Fitting

데이터의 일반적인 패턴을 학습하는 것 이외에도, 모델은 각 데이터의 유일한 노이즈의 특징을 학습한다.

## Model Tuning

![Part%20%E2%85%A0%20General%20Strategies%2094e175df27a04efb96904900a506f65f/Untitled%2010.png](Part%20%E2%85%A0%20General%20Strategies%2094e175df27a04efb96904900a506f65f/Untitled%2010.png)

KNN이 2 class를 분류할 때 동점을 피하기 위해 파라미터 후보를 홀수로 설정하자.

## Data Splitting

데이터가 크지 않다면 test set 사용 x

 

데이터 나누는 법 

- nonrandom

    1. 특정 환자군을 사용해 모델을 구축한 뒤, 다른 데이터로 모델이 일반화 되는지 확인 

    2. 과거에 존재하는 스팸 기술보다는 최신 스팸 기술을 정확하게 예측할 수 있는가 

- random

    대부분의 경우는 훈련셋과 테스트 셋은 가능한한 동질적이여야 한다. 임의 표집을 사용할 수 있다.

- simple random sampling: 클래스 불균형인 경우 주의한다.
- startified random sampling: 만약 outcome이 연속이면 적당히 subgroup으로 나눈 다음 각 그룹 내에서 랜덤 샘플링
- maximum dissimilarity sampling

    두 샘플 간 비유사도는 수많은 방법으로 측정될 수 있다. 가장 간단한 방법은 두 샘플의 예측 변수 사이의 거리를 이용하는 것이다.

    알고리즘

    1. 단 하나의 초기 샘플을 정해서 나머지 샘플과 비유사도 계산
    2. 가장 비유사도가 높은 샘플을 test set에 추가
    3. 두 개 이상의 샘플부터는 test set 각각의 샘플들과 할당되지 않은 하나의 샘플의 비유사도를 계산한다. 그리고나서 평균을 취한다. 평균 비유사도 중에서 최댓값에 대응하는 샘플을 test set에 추가한다. 
    4. 원하는 test set의 크기가 될 때까지 진행한다. 

## Resampling Techniques

### k-fold Cross-Validation

k가 커질수록 bias가 감소하고 분산이 커진다.

k=10은 LOOCV과 비슷한 결과를 내지만 계산이 효율적이다.

선형회귀모델에서는 leave-one-out error rate에 근사가 되는 공식이 있다.

generalized cross-validation(GCV) 통계량

$$\text{GCV} = \frac{1}{n}\sum_{i=1}^n \left( \frac{y_i - \hat y_i}{1 - \frac{df}{n}} \right) ^2$$

- repeated k-fold CV를 사용할 때 보통 반복횟수 > k 이다.
- bootstrap 방법은 평균적으로 63.2%의 데이터가 적어도 한 번 사용된다. 여기서 사용되지 않은 샘플들을 "out-of-bag" sample이라고 한다. bootstrap방법은 2-fold CV와 비슷한 bias를 가진다. 하지만 training set의 크기가 크면 이 bias는 큰 문제가 되지는 않는다. 참고로 bias를 제거하기 위한 "632 method"가 존재한다.

## Choosing Final Tuning Parameters

one-standard error방법: 가장 좋은 정확도 - s.e 보다 큰 정확도를 가지는 파라미터 중에서 가장 단순한 파라미터 선택 

tolerance: ???

## Data Splitting Recommendations

- 표본 크기 작다: repeated 10-fold CV
- 표본크기 크다: 10-fold CV
- 정확한 성능 추정이 아니라 모델 간 비교: 낮은 분산을 가지는 bootstrap

파라미터 튜닝을 하는 동안 모델 성능 추정을 하면 잠재적으로 "optimization bias"가 생기지만 큰 데이터셋에서는 이러한 편향이 줄어든다.

## Choosing between Models

1. 가장 해석이 어렵지만 성능이 좋은 boosted tree, SVM
2. MARS, PLS, GAM, naive Bayes
3. 가장 단순하지만 좀 더 복잡한 모델의 성능에 근접하는 모델

만약 repeated 10-fold CV를 사용했을 경우 paired t-test를 사용해서 모델 성능 차이를 검정할 수 있다. 

## Exercises

테스트 셋 크기가 작으면 LOOCV

boxcox, center, scale, knnimpute, nearzerovar, highly correlated predictors