# GNSS + IMU 기반 차량 위치 추정 (Extended Kalman Filter)

## 프로젝트 배경

자율주행 차량의 위치 추정(Localization)은 경로 계획, 제어 등 모든 상위 모듈의 기반이 되는 핵심 문제입니다. 이 프로젝트는 KITTI Raw Dataset의 OXTS(GPS/IMU) 데이터를 활용하여,

- **IMU 단독 (Dead Reckoning)**
- **GNSS 단독**
- **Extended Kalman Filter (EKF) 기반 GNSS+IMU 융합**

세 가지 방식의 위치 추정 성능을 직접 비교하고, 각 센서의 한계와 이를 보완하는 센서 융합의 필요성을 실험적으로 검증하는 것을 목표로 합니다.

특히 필터 로직을 라이브러리에 의존하지 않고 NumPy로 직접 구현함으로써, EKF의 예측(Prediction)-보정(Update) 구조와 공분산 전파 과정을 명확히 이해하고자 했습니다.

---

## 데이터셋

- **KITTI Raw Data** (OXTS GPS/IMU 포함)
  - 시퀀스: `[사용한 시퀀스 이름 기재, 예: 2011_09_26_drive_0005]`
  - 포함 데이터: 위도/경도/고도(GNSS), 3축 가속도·각속도(IMU), Ground Truth 궤적

---

## 상태공간 및 EKF 수식

### 상태 벡터

$$
x = [x, y, \psi, v]^T
$$

- $x, y$ : 2D 평면상의 위치
- $\psi$ : yaw (헤딩 각)
- $v$ : 속도

### 예측 단계 (Prediction, IMU 기반)

IMU의 가속도 $a$와 각속도 $\omega$를 이용한 상태 전이 모델:

$$
\begin{aligned}
x_{k+1} &= x_k + v_k \cos(\psi_k) \cdot \Delta t \\
y_{k+1} &= y_k + v_k \sin(\psi_k) \cdot \Delta t \\
\psi_{k+1} &= \psi_k + \omega_k \cdot \Delta t \\
v_{k+1} &= v_k + a_k \cdot \Delta t
\end{aligned}
$$

공분산 예측: $P_{k+1|k} = F_k P_k F_k^T + Q$

### 업데이트 단계 (Update, GNSS 기반)

GNSS로부터 관측된 $(x, y)$를 이용하여 상태를 보정:

$$
z_k = H x_k + v_k, \quad H = \begin{bmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \end{bmatrix}
$$

칼만 이득 계산 후 상태 및 공분산 갱신:

$$
K_k = P_{k|k-1} H^T (H P_{k|k-1} H^T + R)^{-1}
$$
$$
x_k = x_{k|k-1} + K_k (z_k - H x_{k|k-1})
$$
$$
P_k = (I - K_k H) P_{k|k-1}
$$

- $Q$ : 프로세스 노이즈 공분산 (IMU 신뢰도 반영)
- $R$ : 관측 노이즈 공분산 (GNSS 신뢰도 반영)

---

## 프로젝트 구성

| 단계 | 내용 | 목적 |
|---|---|---|
| 1 | IMU 단독 적분 (Dead Reckoning) | 오차 누적 현상 확인 |
| 2 | GNSS 단독 좌표 플롯 | 노이즈로 인한 튐 현상 확인 |
| 3 | EKF 기반 GNSS+IMU 융합 | 두 센서의 상호 보완 확인 |
| 4 | 정량 평가 및 시각화 | RMSE 비교, 궤적 비교 플롯 |

---

## 실험 결과

### 궤적 비교

*(3가지 방법 — IMU만 / GNSS만 / EKF 융합 — 을 Ground Truth와 함께 한 그래프에 플롯)*

```
[trajectory_comparison.png 삽입 위치]
```

- IMU 단독: 초반에는 GT와 유사하지만 시간이 지날수록 오차가 누적되어 궤적이 크게 벗어남
- GNSS 단독: 전반적인 궤적은 따라가지만 지점별로 노이즈에 의한 튐 발생
- EKF 융합: IMU의 단기 정확도와 GNSS의 장기 안정성을 결합하여 가장 GT에 근접한 부드러운 궤적 생성

### RMSE 비교

| 방법 | RMSE (m) |
|---|---|
| IMU 단독 (Dead Reckoning) | `[값 입력]` |
| GNSS 단독 | `[값 입력]` |
| EKF 융합 | `[값 입력]` |

---

## 배운 점

- IMU 적분 오차는 시간에 대해 누적되므로, 짧은 구간에서는 정확하지만 장시간 사용 시 발산한다는 것을 직접 확인함
- GNSS는 절대 위치 기준을 제공하지만 순간적인 노이즈에 취약하며, 단독 사용 시 궤적이 불안정함
- EKF는 두 센서의 상호 보완적 특성(IMU: 고주파 정확도, GNSS: 저주파 안정성)을 공분산 기반으로 자동 가중하여 최적으로 융합함
- $Q$, $R$ 공분산 값의 튜닝이 필터 성능에 큰 영향을 미치며, 두 값의 상대적 크기가 "IMU를 더 믿을지 GNSS를 더 믿을지"를 결정한다는 점을 체감함
- 혈압 추정 연구에서 다루었던 다중 신호 융합(Bi-GRU 기반 특징 융합)과 구조적으로 유사하며, 이번에는 확률적 모델(칼만 필터) 기반의 명시적 융합 방식을 경험함

## 한계 및 개선 방향

- 현재 모델은 등속/등각속도 가정을 사용한 단순 운동 모델로, 급격한 조향이나 가감속 구간에서는 오차가 커질 수 있음
- GNSS 관측 노이즈를 고정값으로 가정했으나, 실제로는 위성 수신 상태에 따라 동적으로 조정하는 Adaptive EKF로 확장 가능
- 향후 UKF(Unscented Kalman Filter) 또는 Particle Filter와의 성능 비교, 혹은 LiDAR/카메라 기반 관측치 추가 융합으로 확장 가능

---

## 기술 스택

- Python, NumPy (EKF 로직 직접 구현)
- Matplotlib (궤적 시각화)
- Dataset: [KITTI Raw Data](http://www.cvlibs.net/datasets/kitti/raw_data.php)

## 실행 방법

```bash
git clone [repo url]
cd [repo name]
pip install -r requirements.txt
python main.py --sequence [시퀀스명]
```

---

## 참고 자료

- KITTI Dataset: http://www.cvlibs.net/datasets/kitti/
- Probabilistic Robotics (Thrun et al.) - EKF 이론 참고
