from IPython.display import HTML, Image, display
from datetime import timedelta
from Code import loadData, plottingTools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import norm
from math import log, sqrt, pi, exp
import scipy.stats as stats

i = complex(0, 1)
n = norm.pdf
N = norm.cdf

def Remove(duplicate):
    final_list = []
    for num in duplicate:
        if num not in final_list:
            final_list.append(num)
    return final_list

############################################################################################
def bsm_price(option_type, sigma, s, k, r, T, q):
    # calculate the bsm price of European call and put options
    sigma = float(sigma)
    d1 = (np.log(s / k) + (r - q + sigma ** 2 * 0.5) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'c':
        price = np.exp(-r*T) * (s * np.exp((r - q)*T) * stats.norm.cdf(d1) - k *  stats.norm.cdf(d2))
        return price
    elif option_type == 'p':
        price = np.exp(-r*T) * (k * stats.norm.cdf(-d2) - s * np.exp((r - q)*T) *  stats.norm.cdf(-d1))
        return price
    else:
        print('No such option type %s') %option_type


def implied_vol(option_type, option_price, s, k, r, T, q):
    # apply bisection method to get the implied volatility by solving the BSM function
    precision = 0.00001
    upper_vol = 500.0
    max_vol = 500.0
    min_vol = 0.0001
    lower_vol = 0.0001
    iteration = 0

    while 1:
        iteration += 1
        mid_vol = (upper_vol + lower_vol) / 2.0
        price = bsm_price(option_type, mid_vol, s, k, r, T, q)
        if option_type == 'c':

            lower_price = bsm_price(option_type, lower_vol, s, k, r, T, q)
            if (lower_price - option_price) * (price - option_price) > 0:
                lower_vol = mid_vol
            else:
                upper_vol = mid_vol
            if abs(price - option_price) < precision: break
            if mid_vol > max_vol - 5:
                mid_vol = 0.000001
                break
        #             print("mid_vol=%f" %mid_vol)
        #             print("upper_price=%f" %lower_price)

        elif option_type == 'p':
            upper_price = bsm_price(option_type, upper_vol, s, k, r, T, q)

            if (upper_price - option_price) * (price - option_price) > 0:
                upper_vol = mid_vol
            else:
                lower_vol = mid_vol
            #             print("mid_vol=%f" %mid_vol)
            #             print("upper_price=%f" %upper_price)
            if abs(price - option_price) < precision: break
            if iteration > 50: break

    return mid_vol

# #############################################################################################"

def d1(S, K, T, r, sigma):
    return (log(S / K) + (r + sigma ** 2 / 2.) * T) / sigma * sqrt(T)


def d2(S, K, T, r, sigma):
    return d1(S, K, T, r, sigma) - sigma * sqrt(T)


def bs_call(S, K, T, r, sigma):

    return S * norm.cdf(d1(S, K, T, r, sigma)) - K * exp(-r * T) * norm.cdf(d2(S, K, T, r, sigma))

# ###########################################################################


def bs_vega(cp_flag,S,K,T,r,v,q=0.0):
    d1 = (log(S/K)+(r+v*v/2.)*T)/(v*sqrt(T))
    return S * sqrt(T)*n(d1)


def bs_price(cp_flag,S,K,T,r,v,q=0.0):
    d1 = (log(S/K)+(r+v*v/2.)*T)/(v*sqrt(T))
    d2 = d1-v*sqrt(T)
    if cp_flag == 'c':
        price = S*exp(-q*T)*N(d1)-K*exp(-r*T)*N(d2)
    else:
        price = K*exp(-r*T)*N(-d2)-S*exp(-q*T)*N(-d1)
    return price


def find_vol(target_value, call_put, S, K, T, r):
    MAX_ITERATIONS = 100
    PRECISION = 1.0e-5

    sigma = 0.5
    for i in range(0, MAX_ITERATIONS):
        price = bs_price(call_put, S, K, T, r, sigma)
        vega = bs_vega(call_put, S, K, T, r, sigma)

        price = price
        diff = target_value - price  # our root

        #print i, sigma, diff

        if (abs(diff) < PRECISION):
            return sigma
        sigma = sigma + diff/vega # f(x) / f'(x)

    # value wasn't found, return best guess so far
    return sigma


# #############################################################################################"


def fHeston(s, St, K, r, T, sigma, kappa, theta, volvol, rho):
    # To be used a lot
    prod = rho * sigma * i * s

    # Calculate d
    d1 = (prod - kappa) ** 2
    d2 = (sigma ** 2) * (i * s + s ** 2)
    d = np.sqrt(d1 + d2)

    # Calculate g
    g1 = kappa - prod - d
    g2 = kappa - prod + d
    g = g1 / g2

    # Calculate first exponential
    exp1 = np.exp(np.log(St) * i * s) * np.exp(i * s * r * T)
    exp2 = 1 - g * np.exp(-d * T)
    exp3 = 1 - g
    mainExp1 = exp1 * np.power(exp2 / exp3, -2 * theta * kappa / (sigma ** 2))

    # Calculate second exponential
    exp4 = theta * kappa * T / (sigma ** 2)
    exp5 = volvol / (sigma ** 2)
    exp6 = (1 - np.exp(-d * T)) / (1 - g * np.exp(-d * T))
    mainExp2 = np.exp((exp4 * g1) + (exp5 * g1 * exp6))
    return (mainExp1 * mainExp2)


def priceHestonMid(St, K, r, T, params):
    u1 = 0.5
    sigma = params[0]
    kappa = params[1]
    theta = params[2]
    volvol = params[3]
    rho = params[4]

    P, iterations, maxNumber = 0, 1000, 100
    ds = maxNumber / iterations

    element1 = 0.5 * (St - K * np.exp(-r * T))

    # Calculate the complex integral
    # Using j instead of i to avoid confusion
    for j in range(1, iterations):
        s1 = ds * (2 * j + 1) / 2
        s2 = s1 - i

        numerator1 = fHeston(s2, St, K, r, T, sigma, kappa, theta, volvol, rho)
        numerator2 = K * fHeston(s1, St, K, r, T, sigma, kappa, theta, volvol, rho)
        denominator = np.exp(np.log(K) * i * u1) * i * u1

        P += ds * (numerator1 - numerator2) / denominator

    element2 = P / np.pi

    return np.real((element1 + element2))


def errorPlot(rmse, l1, relative_rmse, relative_l1):
    cells = [[rmse.mean(), l1.mean(), relative_rmse.mean(), relative_l1.mean()],
             [rmse.max(), l1.max(), relative_rmse.max(), relative_l1.max()],
             [rmse.idxmax().strftime('%Y-%m-%d'), l1.idxmax().strftime('%Y-%m-%d'),
              relative_rmse.idxmax().strftime('%Y-%m-%d'), relative_l1.idxmax().strftime('%Y-%m-%d')]]
    columns = ["Absolute RMSE error", "Absolute L1 error", "Relative RMSE error", "Relative L1 error"]
    index = ["Daily average error", "Worst dataset error", "Worst dataset day"]
    print("")
    display(HTML(pd.DataFrame(cells, index=index, columns=columns).to_html()))
    print("")

    title = None
    plottingTools.historyPlot(rmse, title=(title + " RMSE ") if (title is not None) else " RMSE ")
    plottingTools.historyPlot(relative_rmse,
                              title=(title + " Relative RMSE ") if (title is not None) else " Relative RMSE ")


def errorPlotSmile(rmse, bench_rmse, relative_rmse, relative_bench_rmse, batch, coordinates, worstDayRef, model_vol,
                   interp_vol, rnkMat):
    cells = [[rmse.mean(), bench_rmse.mean(), relative_rmse.mean(), relative_bench_rmse.mean()],
             [rmse.max(), bench_rmse.max(), relative_rmse.max(), relative_bench_rmse.max()],
             [rmse.idxmax().strftime('%Y-%m-%d'), bench_rmse.idxmax().strftime('%Y-%m-%d'),
              relative_rmse.idxmax().strftime('%Y-%m-%d'), relative_bench_rmse.idxmax().strftime('%Y-%m-%d')]]
    columns = ["Interpolation-Absolute RMSE error", "Stochastic-Absolute RMSE error",
               "Interpolation-Relative RMSE error", "Stochastic-Relative RMSE error"]
    index = ["Daily average error", "Worst dataset error", "Worst dataset day"]
    print("")
    display(HTML(pd.DataFrame(cells, index=index, columns=columns).to_html()))
    print("")

    # Plot smile per maturity
    expiries = coordinates.applymap(lambda x: x[0]) if type(coordinates) == type(
        pd.DataFrame()) else coordinates.map(lambda x: x[0])
    expirieslist = Remove(expiries)
    df2 = expiries.rank()
    ListRank = df2.rank(method='dense')

    for ttm in expirieslist:
        indx = [x for x in expiries.index if expiries[x] == ttm][0]
        rnk = ListRank[indx]
        market_vols = []
        calib_vols = []
        interp_vols = []
        strikes = []
        for i, t in enumerate(coordinates):
            if t[0] == ttm:
                try:
                    calib_vols.append(model_vol[worstDayRef, t[0], t[1]])
                    strikes.append(t[1])
                    market_vols.append(batch.loc[coordinates.index[i]])
                    cle = (worstDayRef, t[0], t[1])
                    if cle in list(interp_vol.keys()):
                        interp_vols.append(interp_vol[worstDayRef, t[0], t[1]])
                    else:
                        interp_vols.append(None)
                except KeyError:
                    continue

        mse = np.mean(np.square(np.array(market_vols) - np.array(calib_vols))) ** 0.5
        plt.plot(strikes, market_vols, label="Market")
        plt.plot(strikes, calib_vols, label="Heston")
        if rnk in rnkMat:
            plt.plot(strikes, interp_vols, marker='x', label="Linear Interpolation")

        plt.xlabel('Log Moneyness', fontsize=8)
        plt.ylabel('Volatility', fontsize=8)
        plt.title('Maturity: ' + str(round(ttm, 1)) + 'Y' + ' || Mean Square Error: ' + str(round(100 * mse, 1)) + '%',
                  fontsize=10)
        plt.legend(fontsize=8)
        plt.show()


class Heston:

    def __init__(self, rf_rate=0.01, div_rate=0.0):
        # Initialize Model params with dummy values
        self.v_0 = 0.01
        self.kappa = 0.02
        self.theta = 0.5
        self.rho = 0.5
        self.sigma = 0.5

        # Set calendar and other variables
        self.risk_free_rate = rf_rate
        self.dividend_rate = div_rate
        self.model_vol = {}
        self.model_params = {}

    def diagnoseCompression(self, dataList, restoreResults=False):
        self.model_vol = {}
        self.model_params = {}
        rmse = []
        l1 = []
        relative_rmse = []
        relative_l1 = []

        if restoreResults:
            return
            # resCompression = self.readObject("compressionResult")

        else:
            dates = dataList[0].index
            for i, ref_date in enumerate(dates):
                # selectData for calibration
                Selected_dataList = [dataList[0].iloc[i:i + 2], dataList[1].iloc[i:i + 2], dataList[2].iloc[i:i + 2]]

                # compute spot
                fwd = np.array(Selected_dataList[2].loc[ref_date])[0]
                TTM = np.array(Selected_dataList[1].loc[ref_date])[0][0]
                self.spot = fwd / (1 + self.risk_free_rate) ** TTM

                # looping on maturities
                # Generate current day batch
                filteredBatch, filteredBatchCoordinates, filteredBatchFwd = loadData.removePointsWithInvalidCoordinatesWithFwd(
                    Selected_dataList[0].iloc[0], Selected_dataList[1].iloc[0], Selected_dataList[2].iloc[0])
                expiries = filteredBatchCoordinates.applymap(lambda x: x[0]) if type(filteredBatchCoordinates) == type(
                    pd.DataFrame()) else filteredBatchCoordinates.map(lambda x: x[0])
                expirieslist = Remove(np.array(expiries))

                errors = []
                relative_errors = []
                # Filter on maturity get dataList(ttm)
                for k, ttm in enumerate(expirieslist):
                    self.TTM = ttm
                    # Generate data for heston calibration helper
                    TTMfilteredBatch, TTMfilteredBatchCoordinates, TTMfilteredBatchFwd = loadData.FilterOnMaturity(
                        filteredBatch, filteredBatchCoordinates, filteredBatchFwd, ttm)
                    Filtered_dataList = [TTMfilteredBatch, TTMfilteredBatchCoordinates, TTMfilteredBatchFwd]

                    strikes = TTMfilteredBatchCoordinates.applymap(lambda x: 1 / np.exp(x[1])) if type(
                        TTMfilteredBatchCoordinates) == type(pd.DataFrame()) else TTMfilteredBatchCoordinates.map(
                        lambda x: 1 / np.exp(x[1]))
                    self.strikes = strikes * TTMfilteredBatchFwd

                    moneyness = TTMfilteredBatchCoordinates.applymap(lambda x: x[1]) if type(
                        TTMfilteredBatchCoordinates) == type(
                        pd.DataFrame()) else TTMfilteredBatchCoordinates.map(lambda x: x[1])

                    #creating couples strike,vols
                    TTMfilteredData = [(self.strikes.iloc[k],v) for k,v in enumerate(TTMfilteredBatch)]
                    TTMfilteredData = pd.Series(TTMfilteredData, index=TTMfilteredBatch.index)

                    self.BSprices = TTMfilteredData.applymap(lambda x: bs_call(self.spot, x[0], ttm, self.risk_free_rate, x[1])) if type(
                        TTMfilteredBatchCoordinates) == type(
                        pd.DataFrame()) else TTMfilteredData.map(lambda x: bs_call(self.spot, x[0], ttm, self.risk_free_rate, x[1]))

                    # Calibration
                    initial_condition = self.sigma, self.kappa, self.theta, self.v_0, self.rho

                    bounds = ((0, None), (0, None), (0, None), (0, None), (-1, 1))
                    cons = (
                        {'type': 'ineq', 'fun': lambda x: x[0]},
                        {'type': 'ineq', 'fun': lambda x: x[2]},
                        {'type': 'ineq', 'fun': lambda x: x[3]},
                        {'type': 'ineq', 'fun': lambda x: 0.99 - x[4]},
                        {'type': 'ineq', 'fun': lambda x: x[4]+0.99},
                    )

                    #self.Debug = initial_condition

                    def costfunction(params):
                        # params= sigma, kappa, theta, volvol, rho. Order to be respected

                        HestonPrices = np.array([
                            priceHestonMid(self.spot, strike, self.risk_free_rate, self.TTM, params)
                            for strike in self.strikes
                        ])
                        return ((HestonPrices - np.array(self.BSprices)) ** 2).mean() ** .5

                    res = minimize(costfunction, initial_condition, bounds=bounds)

                    # update model param required
                    self.sigma, self.kappa, self.theta, self.v_0, self.rho = res['x']
                    self.new_params = res['x']
                    self.model_params[ref_date, ttm] = res['x']

                    # Generate model volatilities for current day
                    current_day = ref_date

                    for j, v in enumerate(TTMfilteredBatch):
                        s = np.array(self.strikes)[j]
                        m = np.array(moneyness)[j]

                        #Actual_vol = v
                        Actual_vol = np.array(self.BSprices)[j]
                        price = priceHestonMid(self.spot, s, self.risk_free_rate, self.TTM, self.new_params)
                        Calibrated_vol = price

                        #Calibrated_vol = implied_vol('c', price, self.spot, s, self.risk_free_rate, self.TTM, 0)
                        err = Actual_vol - Calibrated_vol

                        # model_vol.append(Calibrated_vol)
                        self.model_vol[current_day, ttm, m] = Calibrated_vol
                        errors.append(err)
                        relative_errors.append(err / Actual_vol)

                rmse.append(np.nanmean(np.square(errors)) ** 0.5)
                l1.append(np.nanmean(np.absolute(errors)))
                relative_rmse.append(np.nanmean(np.square(relative_errors)) ** 0.5)
                relative_l1.append(np.nanmean(np.absolute(relative_errors)))

            self.rmse = pd.Series(rmse, index=dataList[0].index)
            self.l1 = pd.Series(l1, index=dataList[0].index)
            self.relative_rmse = pd.Series(relative_rmse, index=dataList[0].index)
            self.relative_l1 = pd.Series(relative_l1, index=dataList[0].index)
            errorPlot(self.rmse, self.l1, self.relative_rmse, self.relative_l1)

            # graph of worst result
            plottingTools.printDelimiter()
            worstDayRef = self.rmse.idxmax()
            batch = dataList[0].loc[worstDayRef]
            batchcoordinates = dataList[1].loc[worstDayRef]
            filteredBatch, filteredBatchCoordinates = loadData.removePointsWithInvalidCoordinates(batch,
                                                                                                  batchcoordinates)
            worstVolRef = filteredBatch

            # reconstructed surface
            vols = []
            for t in filteredBatchCoordinates:
                try:
                    vols.append(self.model_vol[worstDayRef, t[0], t[1]])
                except KeyError:
                    vols.append(None)

            worstVolPred = pd.Series(vols, filteredBatch.index)
            worstDayERR = worstVolPred - worstVolRef
            plottingTools.plotGrid(worstVolRef, filteredBatchCoordinates, "Worst Day Actual Vol",
                                   plotType="transparent")
            plottingTools.plotGrid(worstVolPred, filteredBatchCoordinates, "Worst Day Predicted Vol",
                                   plotType="transparent")

            plottingTools.plotGrid(worstDayERR, filteredBatchCoordinates, "Worst Day Error", plotType="transparent")

            plottingTools.plotTails([worstVolRef, filteredBatchCoordinates, None, None],
                                    [worstVolPred, filteredBatchCoordinates, None, None])

            # Plot smile per maturity
            expiries = filteredBatchCoordinates.applymap(lambda x: x[0]) if type(filteredBatchCoordinates) == type(
                pd.DataFrame()) else filteredBatchCoordinates.map(lambda x: x[0])
            expirieslist = Remove(np.array(expiries))

            for ttm in expirieslist:
                vols = []
                calib_vols = []
                strikes = []

                for i, t in enumerate(filteredBatchCoordinates):
                    if t[0] == ttm:
                        try:
                            calib_vols.append(self.model_vol[worstDayRef, t[0], t[1]])
                            strikes.append(t[1])
                            vols.append(filteredBatch.loc[filteredBatchCoordinates.index[i]])
                        except KeyError:
                            continue

                mse = np.mean(np.square(np.array(vols) - np.array(calib_vols))) ** 0.5
                plt.plot(strikes, vols, label="Market")
                plt.plot(strikes, calib_vols, label="Heston")
                plt.xlabel('Log Moneyness', fontsize=8)
                plt.ylabel('Volatility', fontsize=8)
                plt.title('Maturity: ' + str(round(ttm, 1)) + 'Y' + ' || Mean Square Error: ' + str(
                    round(100 * mse, 1)) + '%', fontsize=10)
                plt.legend(fontsize=8)
                plt.show()

            # Save Results

            # [self.rmse, self.l1, self.relative_rmse, self.relative_l1]
            return


    def CheckFwd(self, dataList):
        #datalist covers one ref day
        # To be completed
        complete_dataList = loadData.removePointsWithInvalidCoordinatesWithFwd(dataList[0], dataList[1], dataList[2])
