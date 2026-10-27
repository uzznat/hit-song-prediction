"""Builds and executes timeseries/time_series_hot100.ipynb.

    py timeseries/build_notebook.py
"""
import os
import nbformat as nbf
from nbclient import NotebookClient

HERE = os.path.dirname(os.path.abspath(__file__))
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))

md(r"""
# Time series components and forecasting, on Billboard Hot 100 data

Lior Shalon, Nathanael Uzzan

This notebook follows the reading plan we were given, and applies each step to data from our hit-prediction
project (the weekly Billboard Hot 100, August 1958 to July 2026, 354,500 chart entries):

1. **Components of a time series**: trend, seasonality, residual.
2. **How each component is modelled**: moving averages and regression for the trend, seasonal dummies and
   **Fourier terms** for seasonality, autocorrelation tests for the residual.
3. **Forecasting methods**: simple baselines and exponential smoothing.
4. **ARIMA** and its seasonal and regression variants, then **machine-learning** forecasters.

Reading used: GeeksforGeeks, *Components of Time Series Data*; M. Ibrahim, *An Introduction to Time Series
Analysis and Forecasting* (Weights & Biases); R. J. Hyndman and G. Athanasopoulos, *Forecasting: Principles and
Practice*, 3rd ed. (otexts.com/fpp3); R. B. Cleveland et al., *STL: A seasonal-trend decomposition procedure
based on loess* (J. Official Statistics, 1990); G. Box and G. Jenkins, *Time Series Analysis: Forecasting and
Control*.

Two series are built from the chart:

* **Series A, debuts**: the mean number of songs entering the Hot 100 for the first time per chart week, by
  month. It has a strong trend and weak seasonality, and is the series we forecast.
* **Series B, holiday songs**: the number of holiday songs (Christmas, Santa, Jingle, ...) on the chart each
  week. It is almost pure seasonality, and is where Fourier analysis is easiest to see.
""")

code(r"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.seasonal import STL, seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from scipy import stats
import os
warnings.filterwarnings("ignore")      # statsmodels resets its own filters on import
os.makedirs("figures", exist_ok=True)
plt.rcParams.update({"figure.figsize": (11, 3.6), "axes.grid": True, "grid.alpha": .3})
SEED = 42

d = pd.read_csv("../data/hot100.csv", parse_dates=["chart_week"])
weeks = np.sort(d.chart_week.unique())
print(f"{len(d):,} chart entries, {len(weeks):,} weeks, "
      f"{d.chart_week.min():%Y-%m-%d} to {d.chart_week.max():%Y-%m-%d}")
""")

md(r"""
## 0. Building the two series

A *debut* is the first week a (title, performer) pair appears anywhere in the chart history. Months contain
four or five chart weeks, so summing debuts per month would create a false "seasonality" (months with five
Saturdays look busier). We therefore take the **mean per chart week** within each month. We keep complete years,
1959 to 2025.
""")

code(r"""
first = d.groupby(["title", "performer"]).chart_week.min()
weekly_debuts = first.value_counts().reindex(weeks, fill_value=0).sort_index()
A = weekly_debuts.resample("MS").mean()["1959":"2025"]
A.name = "debuts per week"

HOLIDAY = ("Christmas|Santa|Jingle|Sleigh|Rudolph|Mistletoe|Navidad|Winter Wonderland|Holly Jolly|"
           "Silent Night|Snowman")
hol = d[d.title.str.contains(HOLIDAY, case=False, regex=True)]
B = hol.groupby("chart_week").size().reindex(weeks, fill_value=0).sort_index()
B.name = "holiday songs on chart"

fig, ax = plt.subplots(2, 1, figsize=(11, 6))
A.plot(ax=ax[0], lw=.8, title="Series A: mean weekly debuts on the Hot 100, by month")
B.plot(ax=ax[1], lw=.6, color="C3", title="Series B: holiday songs on the Hot 100, weekly")
plt.tight_layout(); plt.savefig("figures/01_series.png", dpi=120); plt.show()
print(f"A: {len(A)} months, mean {A.mean():.2f}   "
      f"B: {len(B)} weeks, {int((B > 0).sum())} weeks with a holiday song")
print("B, highest weekly count per year:", B.groupby(B.index.year).max()[[1960, 1980, 2000, 2010, 2017, 2018, 2020, 2025]].to_dict())
""")

md(r"""
## 1. Components of a time series

A series $y_t$ is described as the combination of

* a **trend** $T_t$: the slow, long-run movement;
* a **seasonal** component $S_t$: a pattern that repeats with a fixed, known period $m$ (12 for monthly data,
  about 52.18 for weekly data);
* a **residual** (remainder, irregular) $R_t$: what is left. If the decomposition is good, it should look like
  noise with no structure left.

Some texts add a **cyclical** component (rises and falls without a fixed period, such as business cycles); in
practice it is usually absorbed into the trend ("trend-cycle").

The components combine **additively**, $y_t = T_t + S_t + R_t$, when the size of the seasonal swings does not
depend on the level of the series, or **multiplicatively**, $y_t = T_t \times S_t \times R_t$, when it grows with
the level. A multiplicative model becomes additive after a log transform:
$\log y_t = \log T_t + \log S_t + \log R_t$. Series B is a clear multiplicative case: the highest weekly count of holiday songs was 2 in
2000 and 2010, 7 in 2017, 16 in 2018 and 33 in 2025, once streaming counts entered the chart.

### 1.1 Classical decomposition and STL

**Classical decomposition** estimates the trend with a centred $2\times12$ moving average, averages the detrended
values month by month to get $S_t$, and takes the rest as $R_t$. It loses six months at each end and assumes a
seasonal pattern that never changes.

**STL** (Seasonal-Trend decomposition using Loess) estimates trend and seasonality with local regressions, lets
the seasonal pattern evolve slowly, and with `robust=True` is not distorted by outliers.

The **strength** of each component (Hyndman & Athanasopoulos, Sec. 4.3) is
$F_T = \max\!\bigl(0, 1 - \tfrac{\operatorname{Var}(R)}{\operatorname{Var}(T+R)}\bigr)$ and
$F_S = \max\!\bigl(0, 1 - \tfrac{\operatorname{Var}(R)}{\operatorname{Var}(S+R)}\bigr)$: 0 means no such
component, 1 means the component explains everything.
""")

code(r"""
classic = seasonal_decompose(A, model="additive", period=12)
stl = STL(A, period=12, robust=True).fit()

fig, ax = plt.subplots(4, 1, figsize=(11, 8), sharex=True)
A.plot(ax=ax[0], lw=.8, title="observed")
stl.trend.plot(ax=ax[1], label="STL"); classic.trend.plot(ax=ax[1], ls="--", label="classical 2x12 MA")
ax[1].set_title("trend"); ax[1].legend()
stl.seasonal.plot(ax=ax[2], lw=.7, title="seasonal (STL)")
stl.resid.plot(ax=ax[3], lw=.6, title="residual (STL)")
plt.tight_layout(); plt.savefig("figures/02_decomposition.png", dpi=120); plt.show()

strength = lambda comp, r: max(0.0, 1 - r.var() / (comp + r).var())
F_T, F_S = strength(stl.trend, stl.resid), strength(stl.seasonal, stl.resid)
print(f"Series A: trend strength F_T = {F_T:.2f}, seasonal strength F_S = {F_S:.2f}")
print("Mean seasonal effect by month (debuts per week):")
print(stl.seasonal.groupby(A.index.month).mean().round(2).to_frame("S").T.to_string())
print("Trend: 1959 {:.1f}, minimum {:.1f} in {}, 2025 {:.1f}".format(
    stl.trend["1959"].mean(), stl.trend.min(), stl.trend.idxmin().year, stl.trend["2025"].mean()))
""")

md(r"""
**Reading the result.** The trend dominates ($F_T = 0.72$): about 11 debuts per week in 1959, a minimum of 5.5
in 2001 (songs stayed on the chart longer, so fewer new ones entered), and back to about 11 in 2025 after the
chart started counting streams. Seasonality is weak ($F_S = 0.11$): January is the quietest month for new entries (release schedules stop over
the holidays). The residual still contains spikes. The largest, in December 1998, is a chart-rule change: from
the chart of 5 December 1998, songs not released as singles became eligible, and 60 songs debuted in one week.
The spikes of the 2020s are album-release weeks, when streaming puts most of a star's new album on the chart at
once.

## 2. Modelling each component

### 2.1 Trend

* **Moving average**: smooths out the season and the noise; simple, but gives no model to extend into the future.
* **Regression on time**: $T_t = \beta_0 + \beta_1 t (+ \beta_2 t^2 \dots)$. Easy to extrapolate, but a
  polynomial goes wrong quickly outside the data.
* **Local regression (loess)**: what STL uses; flexible, but not meant for extrapolation either.
* **Differencing**: instead of modelling the trend, remove it: $y'_t = y_t - y_{t-1}$. This is the "I"
  (integrated) in ARIMA.
""")

code(r"""
t = np.arange(len(A))
fits = {deg: np.polyval(np.polyfit(t, A.values, deg), t) for deg in (1, 2, 3)}
ax = A.plot(lw=.6, color="grey", label="observed", title="Trend models for series A")
A.rolling(12, center=True).mean().plot(ax=ax, lw=2, label="12-month moving average")
for deg, f in fits.items():
    ax.plot(A.index, f, label=f"polynomial, degree {deg}")
ax.legend(ncol=3, fontsize=8); plt.tight_layout(); plt.savefig("figures/03_trend.png", dpi=120); plt.show()
for deg, f in fits.items():
    print(f"polynomial degree {deg}: residual std {np.std(A.values - f):.2f}")
print(f"12-month moving average: residual std {np.nanstd(A.values - A.rolling(12, center=True).mean().values):.2f}")
""")

md(r"""
### 2.2 Seasonality: dummies and Fourier terms

**Seasonal dummies** give each month its own coefficient ($m-1$ parameters). That works for $m=12$, but for
weekly data ($m \approx 52.18$, not even an integer) it needs 51 parameters and cannot handle the fractional
period.

**Fourier terms** describe any periodic pattern of period $m$ as a sum of sines and cosines:
$$S_t = \sum_{k=1}^{K} \Bigl[a_k \sin\!\Bigl(\frac{2\pi k t}{m}\Bigr) + b_k \cos\!\Bigl(\frac{2\pi k t}{m}\Bigr)\Bigr].$$
With $K$ harmonics we need only $2K$ parameters, $m$ may be fractional, and a small $K$ gives a smooth pattern,
a larger $K$ a sharper one. $K$ is chosen by AIC or by cross-validation.

**Where is the period?** The **periodogram**, computed with the **Fast Fourier Transform (FFT)**, shows how much
of the variance of the series sits at each frequency. A seasonal series has a peak at frequency $1/m$ and at its
harmonics $2/m, 3/m, \dots$ We apply it to series B (weekly), after a log transform because the season is
multiplicative, and after removing the slow trend with a two-year moving average.
""")

code(r"""
yB = np.log1p(B.values.astype(float))
yB_d = yB - pd.Series(yB).rolling(105, center=True, min_periods=1).mean().values
n = len(yB_d)
power = np.abs(np.fft.rfft(yB_d - yB_d.mean())) ** 2 / n
freq = np.fft.rfftfreq(n, d=1.0)                       # cycles per week
from scipy.signal import find_peaks
pk, _ = find_peaks(power, distance=20)                 # one index per peak, not its neighbouring bins
top = pk[np.argsort(power[pk])[::-1][:6]]

fig, ax = plt.subplots()
ax.plot(freq[1:], power[1:], lw=.7)
for k in top[:4]:
    ax.annotate(f"{1 / freq[k]:.1f} weeks", (freq[k], power[k]), textcoords="offset points",
                xytext=(5, 5), fontsize=8)
ax.set_xlim(0, .2); ax.set_xlabel("frequency (cycles per week)"); ax.set_ylabel("power")
ax.set_title("Periodogram (FFT) of series B, log scale, detrended")
plt.tight_layout(); plt.savefig("figures/04_periodogram.png", dpi=120); plt.show()
print("Strongest periods (weeks):", [round(float(1 / freq[k]), 1) for k in top])
""")

md(r"""
The peaks fall at about **52 weeks** and its harmonics (26, 17.4, 13 weeks, ...): the FFT recovers the yearly
cycle without being told. The harmonics are there because the holiday pattern is not a smooth sine wave but a
short, sharp peak in December; reproducing a sharp peak needs several harmonics.

Now we fit a **Fourier regression** to series B: $\log(1+y_t) = $ trend (a cubic in time)
$+ \sum_{k\le K}$ Fourier terms with $m = 365.25/7 \approx 52.18$, and pick $K$ by AIC.
""")

code(r"""
M_WEEK = 365.25 / 7
tt = np.arange(len(B))
def fourier(t, m, K):
    return np.column_stack([f(2 * np.pi * k * t / m) for k in range(1, K + 1) for f in (np.sin, np.cos)])
trend_X = np.column_stack([(tt / len(B)) ** p for p in (1, 2, 3)])

res = {}
for K in range(1, 21):
    X = sm.add_constant(np.column_stack([trend_X, fourier(tt, M_WEEK, K)]))
    res[K] = sm.OLS(yB, X).fit()
aic = pd.Series({K: r.aic for K, r in res.items()})
K_best = int(aic.idxmin())
print("AIC by K:", aic.round(0).astype(int).to_dict())
print(f"best K = {K_best} harmonics ({2 * K_best} parameters, instead of 51 weekly dummies); "
      f"R^2 = {res[K_best].rsquared:.2f} (trend only: "
      f"{sm.OLS(yB, sm.add_constant(trend_X)).fit().rsquared:.2f})")

fit = pd.Series(np.expm1(res[K_best].fittedvalues), index=B.index)

# The amplitude of the season changed with streaming: fit the same model on the streaming era alone.
recent = B.index >= "2018-07-01"
yR, tR = yB[recent], tt[recent]
resR = {K: sm.OLS(yR, sm.add_constant(np.column_stack([(tR - tR[0]) / len(tR), fourier(tR, M_WEEK, K)]))).fit()
        for K in range(1, 27)}                       # at most m/2 = 26 distinct harmonics
K_R = min(resR, key=lambda K: resR[K].bic)     # BIC: AIC keeps adding harmonics on 420 weeks
fitR = pd.Series(np.expm1(resR[K_R].fittedvalues), index=B.index[recent])
print(f"streaming era (2018-07 on): best K by BIC = {K_R}, R^2 = {resR[K_R].rsquared:.2f}")

fig, ax = plt.subplots(1, 2, figsize=(11, 3.6), gridspec_kw={"width_ratios": [2, 1]})
ax[0].plot(B.index, yB, lw=.5, color="grey", label="observed, log(1+y)")
ax[0].plot(B.index, res[K_best].fittedvalues, color="C3", label=f"whole period, K={K_best}")
ax[0].legend(fontsize=8); ax[0].set_title("Series B (log scale) and a Fourier fit on the whole period")
B["2018-07":].plot(ax=ax[1], lw=.8, color="grey", label="observed")
fit["2018-07":].plot(ax=ax[1], color="C3", label="whole-period fit")
fitR.plot(ax=ax[1], color="C2", label=f"streaming-era fit, K={K_R}")
ax[1].legend(fontsize=7); ax[1].set_title("original scale, 2018-2026")
plt.tight_layout(); plt.savefig("figures/05_fourier_fit.png", dpi=120); plt.show()
""")

md(r"""
On the whole period, the Fourier terms place the peak correctly (every December) but with one fixed amplitude
for 68 years, so they badly underestimate the streaming-era peaks: the amplitude of the season itself changed
around 2018. Fitted on the streaming era alone, the same model follows the peaks closely. Fourier terms assume a
stable seasonal pattern; when the pattern changes, one must either fit on a recent window, let the Fourier
coefficients vary over time, or use a method whose season evolves, such as STL.

### 2.3 Residual

After removing trend and seasonality, the residual should be **white noise**: zero mean, constant variance, no
autocorrelation. We check it with

* the **ACF** (autocorrelation function) and **PACF** (partial autocorrelation): bars outside the shaded band
  indicate structure left in the residual;
* the **Ljung-Box test**: $H_0$ = no autocorrelation up to lag $h$;
* **stationarity tests**: **ADF** ($H_0$: a unit root, i.e. not stationary) and **KPSS** ($H_0$: stationary).
  The two tests have opposite null hypotheses, so they are read together.
""")

code(r"""
r = stl.resid
fig, ax = plt.subplots(1, 2)
plot_acf(r, lags=36, ax=ax[0], title="ACF of the STL residual (A)")
plot_pacf(r, lags=36, ax=ax[1], title="PACF of the STL residual (A)", method="ywm")
plt.tight_layout(); plt.savefig("figures/06_residual_acf.png", dpi=120); plt.show()
print("Ljung-Box on the STL residual:"); print(acorr_ljungbox(r, lags=[12, 24]).round(4))

def stationarity(x, name):
    adf_p = adfuller(x, autolag="AIC")[1]
    kpss_p = kpss(x, regression="c", nlags="auto")[1]
    print(f"{name:28s} ADF p = {adf_p:.3f}   KPSS p = {kpss_p:.3f}")
stationarity(A, "A (raw)")
stationarity(A.diff().dropna(), "A, first difference")
stationarity(r, "A, STL residual")
""")

md(r"""
**Reading the result.** For the raw series, ADF does not reject a unit root and KPSS rejects stationarity: the
series is not stationary. After one difference both tests agree that it is, so one difference ($d=1$) is enough
for ARIMA. The STL residual is close to white noise (Ljung-Box does not reject at lags 12 and 24): trend and
season capture the structure of the series. A decomposition, however, describes the past; it gives no rule to
extend the trend into the future. Forecasting needs a model of how the series moves from one month to the next,
which is what ARIMA and exponential smoothing provide.

## 3. Forecasting methods

**Protocol.** We forecast series A 12 months ahead. A single train/test split can mislead (the lesson of our hit
paper), so we use **rolling-origin evaluation** (time-series cross-validation): for each origin from January 2015
to January 2025, every model is trained on all data before the origin and forecasts the next 12 months; no model
ever sees the future. Scores are averaged over the 11 origins.

**Metrics.** MAE, RMSE, and **MASE** (mean absolute scaled error): the MAE divided by the in-sample MAE of the
seasonal naive forecast. MASE below 1 means more accurate than the seasonal naive method's typical one-year error;
it is scale-free, so it can be compared across series.

**Baselines.**
* *Mean*: the average of the history.
* *Naive*: $\hat y_{T+h} = y_T$ (the last value).
* *Seasonal naive*: $\hat y_{T+h} = y_{T+h-m}$ (same month last year).
* *Drift*: the line from the first to the last observation, extended.

**Exponential smoothing (Holt-Winters, ETS).** Level, trend and season are each updated as weighted averages that
give exponentially less weight to older observations. We use an additive damped trend (the trend flattens out
over the horizon) and additive seasonality.
""")

code(r"""
H, m = 12, 12
origins = pd.date_range("2015-01-01", "2025-01-01", freq="12MS")

def mase_scale(train):
    return np.mean(np.abs(train.values[m:] - train.values[:-m]))

scores = []
def evaluate(name, forecaster):
    for o in origins:
        train, test = A[A.index < o], A[A.index >= o][:H]
        f = np.asarray(forecaster(train, len(test)), float)
        e = test.values - f
        scores.append({"model": name, "origin": o, "MAE": np.mean(np.abs(e)),
                       "RMSE": np.sqrt(np.mean(e ** 2)), "MASE": np.mean(np.abs(e)) / mase_scale(train)})

def summary():
    S = pd.DataFrame(scores)
    return S.groupby("model")[["MAE", "RMSE", "MASE"]].mean().sort_values("MASE").round(3)

def mean_f(tr, h): return np.repeat(tr.mean(), h)
def naive(tr, h): return np.repeat(tr.values[-1], h)
def snaive(tr, h): return np.resize(tr.values[-m:], h)
def drift(tr, h): return tr.values[-1] + (tr.values[-1] - tr.values[0]) / (len(tr) - 1) * np.arange(1, h + 1)
def ets(tr, h):
    return ExponentialSmoothing(tr, trend="add", damped_trend=True, seasonal="add",
                                seasonal_periods=m).fit(optimized=True).forecast(h)

for name, f in [("Mean", mean_f), ("Naive", naive), ("Seasonal naive", snaive), ("Drift", drift),
                ("Holt-Winters (ETS, damped)", ets)]:
    evaluate(name, f)
summary()
""")

md(r"""
## 4. ARIMA and its variants

**ARIMA(p, d, q)** combines three ideas:
* **AR(p)**, autoregression: $y_t$ depends linearly on its own $p$ previous values;
* **I(d)**, integration: the series is differenced $d$ times to make it stationary;
* **MA(q)**, moving average: $y_t$ depends on the $q$ previous forecast errors.

With the backshift operator $B y_t = y_{t-1}$: $\;\phi(B)(1-B)^d y_t = c + \theta(B)\varepsilon_t$.

**Choosing the orders (Box-Jenkins method).** $d$ from the stationarity tests (here $d=1$); then $p$ and $q$ from
the ACF/PACF of the differenced series (a PACF that cuts off after lag $p$ suggests AR($p$); an ACF that cuts off
after lag $q$ suggests MA($q$)), or, as here, by searching a small grid and keeping the lowest AIC. The orders are
chosen once, on the data before the first origin, so the test years never influence them. We search
$p, q \le 2$: larger orders fit the training data marginally better but often fail to converge.

**SARIMA(p,d,q)(P,D,Q)$_m$** adds the same three parts at the seasonal lag $m$ (for example $y_t$ depending on
$y_{t-12}$).

**Dynamic harmonic regression (ARIMA + Fourier).** Seasonality is represented by Fourier terms used as
regressors, and the ARIMA models the remaining autocorrelation. This is the standard approach for long or
fractional seasonal periods, and it connects Sections 2.2 and 4.
""")

code(r"""
train0 = A[A.index < origins[0]]
fig, ax = plt.subplots(1, 2)
plot_acf(train0.diff().dropna(), lags=36, ax=ax[0], title="ACF, differenced series (to 2014)")
plot_pacf(train0.diff().dropna(), lags=36, ax=ax[1], title="PACF, differenced series (to 2014)", method="ywm")
plt.tight_layout(); plt.savefig("figures/07_acf_differenced.png", dpi=120); plt.show()

grid = []
for p in range(3):
    for q in range(3):
        try:
            r_ = SARIMAX(train0, order=(p, 1, q)).fit(disp=False)
            grid.append({"p": p, "q": q, "AIC": r_.aic})
        except Exception:
            pass
grid = pd.DataFrame(grid).sort_values("AIC")
p_best, q_best = int(grid.iloc[0].p), int(grid.iloc[0].q)
print(f"Best non-seasonal order by AIC: ARIMA({p_best},1,{q_best})")
print(grid.head(5).round(1).to_string(index=False))

sgrid = []
for P_ in range(2):
    for Q_ in range(2):
        r_ = SARIMAX(train0, order=(p_best, 1, q_best), seasonal_order=(P_, 0, Q_, 12)).fit(disp=False)
        sgrid.append({"P": P_, "Q": Q_, "AIC": r_.aic})
sgrid = pd.DataFrame(sgrid).sort_values("AIC")
P_best, Q_best = int(sgrid.iloc[0].P), int(sgrid.iloc[0].Q)
print(f"Best seasonal part: (P,D,Q)_12 = ({P_best},0,{Q_best})")

fit0 = SARIMAX(train0, order=(p_best, 1, q_best), seasonal_order=(P_best, 0, Q_best, 12)).fit(disp=False)
print(fit0.summary().tables[1])
print("Ljung-Box on the SARIMA residuals:"); print(acorr_ljungbox(fit0.resid[1:], lags=[12, 24]).round(4))
""")

code(r"""
def fourier_index(idx, K):
    t = (idx.year * 12 + idx.month - 1).values
    return fourier(t, 12, K)

def arima(tr, h):
    return SARIMAX(tr, order=(p_best, 1, q_best)).fit(disp=False).forecast(h)
def sarima(tr, h):
    return SARIMAX(tr, order=(p_best, 1, q_best),
                   seasonal_order=(P_best, 0, Q_best, 12)).fit(disp=False).forecast(h)
def dhr(tr, h, K=2):
    fut = pd.date_range(tr.index[-1] + pd.offsets.MonthBegin(), periods=h, freq="MS")
    r_ = SARIMAX(tr, exog=fourier_index(tr.index, K), order=(p_best, 1, q_best)).fit(disp=False)
    return r_.forecast(h, exog=fourier_index(fut, K))

ARIMA_NAME = f"ARIMA({p_best},1,{q_best})"
SARIMA_NAME = f"SARIMA({p_best},1,{q_best})({P_best},0,{Q_best})12"
for name, f in [(ARIMA_NAME, arima), (SARIMA_NAME, sarima), ("ARIMA + Fourier (K=2)", dhr)]:
    evaluate(name, f)
summary()
""")

md(r"""
## 5. Machine-learning forecasters

A regression model (here gradient boosting and a random forest) does not know about time; forecasting becomes
supervised learning once we build **features from the past**:

* **lags** of the monthly change, $\Delta y_{t-1}, \dots, \Delta y_{t-12}$;
* the **last level** $y_{t-1}$ and a **rolling mean** over the last 12 months;
* **Fourier terms** of the month (seasonality, again).

Two practical points:

1. **Trees cannot extrapolate**: a tree never predicts a value outside the range seen in training, so it cannot
   follow a trend into new territory. We therefore predict the **monthly change** $y_t - y_{t-1}$ (differencing,
   as in ARIMA) and add it back.
2. **Multi-step forecasting**: the model predicts one month ahead; for 12 months we feed each prediction back as
   a lag (the *recursive* strategy). Errors can accumulate along the horizon.
""")

code(r"""
LAGS = 12
def make_xy(series):
    s = pd.Series(series.values, index=series.index)
    dlt = s.diff()
    X = pd.DataFrame({f"d_lag{k}": dlt.shift(k) for k in range(1, LAGS + 1)})
    X["level"] = s.shift(1)
    X["roll12"] = s.shift(1).rolling(12).mean()
    F = fourier_index(s.index, 2)
    for j in range(F.shape[1]):
        X[f"f{j}"] = F[:, j]
    return X, dlt

def ml_forecaster(make_model):
    def f(tr, h):
        X, y = make_xy(tr)
        ok = X.notna().all(axis=1) & y.notna()
        model = make_model().fit(X[ok], y[ok])
        hist = tr.copy()
        out = []
        for _ in range(h):
            nxt = hist.index[-1] + pd.offsets.MonthBegin()
            ext = pd.concat([hist, pd.Series([np.nan], index=[nxt])])
            Xn, _ = make_xy(ext)
            val = hist.values[-1] + model.predict(Xn.iloc[[-1]])[0]
            out.append(val)
            hist = pd.concat([hist, pd.Series([val], index=[nxt])])
        return np.array(out)
    return f

gbm = ml_forecaster(lambda: GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=.03,
                                                      subsample=.8, random_state=SEED))
rf = ml_forecaster(lambda: RandomForestRegressor(n_estimators=300, min_samples_leaf=5,
                                                 random_state=SEED, n_jobs=-1))
for name, f in [("Gradient boosting (lags + Fourier)", gbm), ("Random forest (lags + Fourier)", rf)]:
    evaluate(name, f)

S = pd.DataFrame(scores)
P = S.pivot(index="origin", columns="model", values="MASE")
table = summary()
table["origins better than seasonal naive"] = [int((P[c] < P["Seasonal naive"]).sum()) for c in table.index]
table
""")

md(r"""
### 5.1 Are the differences real?

With 11 origins, some differences between models are noise. We compare the best model with the baselines and
with the runner-up, origin by origin: how many origins it wins, and a paired $t$-test on the 11 MASE values (the
test years of different origins do not overlap).
""")

code(r"""
best = table.index[0]
for other in dict.fromkeys(["Seasonal naive", "Naive", table.index[1]]):
    if other == best:
        continue
    dl = P[other] - P[best]
    print(f"{best}  vs  {other}: MASE {P[best].mean():.3f} vs {P[other].mean():.3f}, "
          f"better at {(dl > 0).sum()}/{len(dl)} origins, paired t-test p = "
          f"{stats.ttest_rel(P[best], P[other]).pvalue:.3f}")
""")

code(r"""
o_last = origins[-1]
tr_last = A[A.index < o_last]
test_last = A[A.index >= o_last][:H]
fig, ax = plt.subplots()
A["2018":].plot(ax=ax, color="black", lw=1, label="observed")
for name, f in [("Seasonal naive", snaive), ("Holt-Winters", ets), ("SARIMA", sarima),
                ("ARIMA + Fourier", dhr), ("Gradient boosting", gbm)]:
    pd.Series(np.asarray(f(tr_last, len(test_last)), float), index=test_last.index).plot(ax=ax, lw=1.4, label=name)
ax.axvline(o_last, color="grey", ls=":")
ax.set_title("12-month forecasts from the last origin (January 2025)"); ax.legend(fontsize=8, ncol=3)
plt.tight_layout(); plt.savefig("figures/08_forecasts.png", dpi=120); plt.show()

cols = list(dict.fromkeys(["Seasonal naive", "Naive", best]))
ax = P[cols].plot(marker="o", title="MASE by forecast origin (lower is better)")
ax.axhline(1, color="grey", ls=":"); plt.tight_layout(); plt.savefig("figures/09_mase_by_origin.png", dpi=120)
plt.show()
""")

md(r"""
## 6. What we take from this

* **Components.** The debut series is mostly trend, with a weak January dip; the holiday series is almost pure,
  multiplicative seasonality whose amplitude jumped when streaming entered the chart.
* **Modelling the components.** The FFT periodogram finds the 52-week period and its harmonics without being told;
  a Fourier regression with a few harmonics replaces 51 weekly dummies. One difference makes the debut series
  stationary, which fixes $d=1$ for ARIMA.
* **Forecasting.** Under rolling-origin evaluation (11 origins, 12-month horizon), ARIMA(1,1,2), ARIMA + Fourier
  and Holt-Winters are the best models (MASE 1.82 to 1.85) and cannot be told apart (paired $p = 0.97$ between the
  first two). They beat the seasonal naive forecast at 10 of 11 origins ($p = 0.006$) and the naive forecast
  ($p = 0.03$). Adding a seasonal ARIMA part does not help (MASE 1.89): with a seasonal strength of 0.11 there is
  little season to model. Every MASE is above 1 because the streaming years (2015 to 2025) move much more than
  the series' history, so all one-year errors are larger than the historical seasonal-naive error.
* **Machine learning.** Gradient boosting and random forests on lag and Fourier features (MASE 1.94 and 1.99) are
  better than the baselines but worse than the classical models. Trees need the target differenced (they cannot
  extrapolate a trend), and on a single series of 800 points they have little to learn that ARIMA does not
  already capture. Their strength appears with many related series or many external regressors.
* **Method.** Two lessons carry over from our hit-prediction paper: evaluate over many origins rather than one
  split, and compare against simple baselines before claiming that a complex model helps.

**Link with our project.** Our paper used the FFT on *audio* signals (spectral descriptors of 30-second previews);
here the same transform finds seasonality in a *chart* signal. Our rolling-origin decay experiment (train on one
decade, test on later ones) is the classification analogue of the rolling-origin evaluation used here.
""")

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
path = os.path.join(HERE, "time_series_hot100.ipynb")
NotebookClient(nb, timeout=1800, kernel_name="python3", resources={"metadata": {"path": HERE}}).execute()
nbf.write(nb, path)
print("written", path)
