import QuantLib as ql
from IPython.display import HTML, Image, display
from datetime import timedelta
from Code import loadData, plottingTools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.optimize import basinhopping
from scipy.optimize import root
from scipy.optimize import differential_evolution

i = complex(0, 1)


def getQLExpiries(coordinates, date):
    expiries = coordinates.applymap(lambda x: date + 365 * timedelta(x[0])) if type(coordinates) == type(
        pd.DataFrame()) else coordinates.map(
        lambda x: date + 365 * timedelta(x[0]))
    Qlexpiries = expiries.applymap(lambda x: ql.Date(x.day, x.month, x.year)) if type(expiries) == type(
        pd.DataFrame()) else expiries.map(
        lambda x: ql.Date(x.day, x.month, x.year))
    return Qlexpiries


def Remove(duplicate):
    final_list = []
    for num in duplicate:
        if num not in final_list:
            final_list.append(num)
    return final_list


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
        plt.ylim([0.25, 0.7])
        plt.title('Maturity: ' + str(round(ttm, 1)) + 'Y' + ' || Mean Square Error: ' + str(round(100 * mse, 1)) + '%',
                  fontsize=10)
        plt.legend(fontsize=8)
        plt.show()


def cost_function_generatorQL(model, helpers, norm=False):
    def cost_function(params):
        params_ = ql.Array(list(params))
        model.setParams(params_)
        error = [h.calibrationError() for h in helpers]
        if norm:
            return np.sqrt(np.sum(np.abs(error)))
        else:
            return error

    return cost_function


class MyBounds(object):
    def __init__(self, xmin=[0., 0.01, 0.01, -1, 0], xmax=[1, 15, 1, 1, 1.0]):
        self.xmax = np.array(xmax)
        self.xmin = np.array(xmin)

    def __call__(self, **kwargs):
        x = kwargs["x_new"]
        tmax = bool(np.all(x <= self.xmax))
        tmin = bool(np.all(x >= self.xmin))
        return tmax and tmin


class Heston:

    def __init__(self, rf_rate=0.01, div_rate=0.0):
        # Initialize Model params with dummy values
        self.v_0 = 0.01
        self.kappa = 0.02
        self.theta = 0.5
        self.rho = 0.5
        self.sigma = 0.5

        # CT
        '''
        self.v_0 = 0.01 or 0.001
        self.kappa = 0.02
        self.theta = 0.5
        self.rho = 0.5
        self.sigma = 0.5
        '''
        # Set calendar and other variables
        self.day_count = ql.Actual365Fixed()
        self.calendar = ql.Japan()
        self.risk_free_rate = rf_rate
        self.dividend_rate = div_rate
        self.dividend_yield = ql.QuoteHandle(ql.SimpleQuote(0.0))
        self.model_vol = {}
        self.model_shift = {}
        self.model_params = {}

    def buildModel(self, ref_date, spot):
        self.v_0 = 0.01
        self.kappa = 0.02
        self.theta = 0.5
        self.rho = 0.5
        self.sigma = 0.5

        self.calculation_date = ql.Date(ref_date.day, ref_date.month, ref_date.year)
        ql.Settings.instance().evaluationDate = self.calculation_date
        self.flat_ts = ql.YieldTermStructureHandle(
            ql.FlatForward(self.calculation_date, self.risk_free_rate, self.day_count))
        self.dividend_ts = ql.YieldTermStructureHandle(
            ql.FlatForward(self.calculation_date, self.dividend_rate, self.day_count))
        process = ql.HestonProcess(self.flat_ts, self.dividend_ts,
                                   ql.QuoteHandle(ql.SimpleQuote(spot)),
                                   self.v_0, self.kappa, self.theta, self.sigma, self.rho)
        model = ql.HestonModel(process)
        return model

    def buildHelper(self, model, ref_date, spot, dataList):
        heston_helpers = []
        self.model = model
        engine = ql.AnalyticHestonEngine(model)

        batch = dataList[0]
        batchcoordinates = dataList[1]
        Fwd = dataList[2]
        filteredBatch, filteredBatchCoordinates, filteredBatchFwd = loadData.removePointsWithInvalidCoordinatesWithFwd(
            batch, batchcoordinates, Fwd)

        data = filteredBatch
        expiration_dates = getQLExpiries(filteredBatchCoordinates, ref_date)
        strikes = filteredBatchCoordinates.applymap(lambda x: 1 / np.exp(x[1])) if type(
            filteredBatchCoordinates) == type(
            pd.DataFrame()) else filteredBatchCoordinates.map(lambda x: 1 / np.exp(x[1]))
        strikes = strikes * filteredBatchFwd

        selected_strikes = strikes
        selected_expiries = expiration_dates
        selected_data = data

        for j, v in enumerate(selected_data):
            #m = np.array(filteredBatchCoordinates)[j][1]

            date = np.array(selected_expiries)[j]
            t = (date - self.calculation_date)
            p = ql.Period(t, ql.Days)
            sigma = v
            s = np.array(selected_strikes)[j]
            helper = ql.HestonModelHelper(p, self.calendar, spot, s,
                                          ql.QuoteHandle(ql.SimpleQuote(sigma)),
                                          self.flat_ts,
                                          self.dividend_ts)
            helper.setPricingEngine(engine)
            heston_helpers.append(helper)

        return heston_helpers, engine

    def diagnoseCompression(self, dataList, restoreResults=False, lm_flag=False):
        self.model_vol = {}
        self.model_shift = {}
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
                spot = fwd / (1 + self.risk_free_rate) ** TTM

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

                    # Generate data for heston calibration helper
                    TTMfilteredBatch, TTMfilteredBatchCoordinates, TTMfilteredBatchFwd = loadData.FilterOnMaturity(
                        filteredBatch, filteredBatchCoordinates, filteredBatchFwd, ttm)
                    Filtered_dataList = [TTMfilteredBatch, TTMfilteredBatchCoordinates, TTMfilteredBatchFwd]
                    expiration_dates = getQLExpiries(TTMfilteredBatchCoordinates, ref_date)

                    strikes = TTMfilteredBatchCoordinates.applymap(lambda x: 1 / np.exp(x[1])) if type(
                        TTMfilteredBatchCoordinates) == type(pd.DataFrame()) else TTMfilteredBatchCoordinates.map(
                        lambda x: 1 / np.exp(x[1]))
                    strikes = strikes * TTMfilteredBatchFwd

                    moneyness = TTMfilteredBatchCoordinates.applymap(lambda x: x[1]) if type(
                        TTMfilteredBatchCoordinates) == type(
                        pd.DataFrame()) else TTMfilteredBatchCoordinates.map(lambda x: x[1])

                    # Build Model & calibration engine
                    model = self.buildModel(ref_date, spot)
                    heston_helpers, engine = self.buildHelper(model, ref_date, spot, Filtered_dataList)

                    # engine = self.buildHelper(model, ref_date, spot, Filtered_dataList)[1]

                    # Calibrate Model
                    initial_condition = list(model.params())
                    bounds = ((0, None), (None, None), (0, None), (-1, 1), (0, None))
                    if lm_flag == True:
                        # lm Quantlib
                        lm = ql.LevenbergMarquardt(1e-8, 1e-8, 1e-8)
                        model.calibrate(heston_helpers, lm, ql.EndCriteria(1000000, 500, 1.0e-8, 1.0e-8, 1.0e-8))

                        ## BASON scipy
                        # mybound = MyBounds()
                        # minimizer_kwargs = {"method": "L-BFGS-B", "bounds": bounds}
                        # minimizer_kwargs = {"method": "L-BFGS-B"}
                        # cost_function = cost_function_generatorQL(model, heston_helpers, norm=True)
                        # sol = basinhopping(cost_function, initial_condition, niter=5, minimizer_kwargs=minimizer_kwargs, stepsize=0.005, accept_test=mybound, interval=10)

                        # differential_evolution scipy
                        # boundsDE = [(-2, 2), (-50, 50), (-10, 10), (-1, 1), (0, 10)]
                        # cost_function = cost_function_generatorQL(model, heston_helpers, norm=True)
                        # sol = differential_evolution(cost_function,bounds=boundsDE) #, maxiter=100

                        # lm scipy
                        # cost_function = cost_function_generatorQL(model, heston_helpers, norm=True)
                        # sol = root(cost_function, initial_condition, method='lm')

                    else:
                        # bounds = ((0, None), (0.02893,0.02894), (0.58897, 0.58898), (0.2334898, 0.2334899), (0.14, 0.14867))
                        cost_function = cost_function_generatorQL(model, heston_helpers, norm=True)
                        sol = minimize(cost_function, initial_condition)  # ,bounds=bounds    #,options={'maxiter': 1000}

                    # update model param required
                    self.theta, self.kappa, self.sigma, self.rho, self.v_0 = model.params()
                    self.model_params[ref_date, ttm] = model.params()
                    self.vol_surf = ql.HestonBlackVolSurface(ql.HestonModelHandle(model), engine.Gatheral)

                    # Generate model volatilities for current day
                    current_day = ref_date
                    current_dayQL = ql.Date(current_day.day, current_day.month, current_day.year)
                    # errors = []
                    # relative_errors = []
                    # model_vol = []
                    for j, v in enumerate(TTMfilteredBatch):
                        date = np.array(expiration_dates)[j]
                        p = (date - current_dayQL) / 365
                        s = np.array(strikes)[j]
                        m = np.array(moneyness)[j]
                        Actual_vol = v

                        try:
                            Calibrated_vol = self.vol_surf.blackVol(date, s)
                        except RuntimeError:
                            continue

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
                # plt.ylim([0.25, 0.7])
                plt.title('Maturity: ' + str(round(ttm, 1)) + 'Y' + ' || Mean Square Error: ' + str(
                    round(100 * mse, 1)) + '%', fontsize=10)
                plt.legend(fontsize=8)
                plt.show()

            # Save Results
            # [self.rmse, self.l1, self.relative_rmse, self.relative_l1]
            return

    def compareInterp(self, ref_date, dataList, outputs_val=None, flag_shift=False):

        rnkMat = []
        rnkMn = []
        filteredData = loadData.FilterOnMaturityMoneneynessRank(dataList[0], dataList[1], dataList[2], rnkMat, rnkMn)
        MoneynessRanks = filteredData[3].loc[ref_date]
        if outputs_val is not None:
            NeuralNet_val = outputs_val.loc[ref_date]

        batch = dataList[0].loc[ref_date]
        batchcoordinates = dataList[1].loc[ref_date]
        batchfwd = dataList[2].loc[ref_date]

        filteredBatch, filteredBatchCoordinates, filteredBatchFwd = loadData.removePointsWithInvalidCoordinatesWithFwd(
            batch, batchcoordinates, batchfwd)

        # 1-Plot smile per maturity
        expiries = filteredBatchCoordinates.applymap(lambda x: x[0]) if type(filteredBatchCoordinates) == type(
            pd.DataFrame()) else filteredBatchCoordinates.map(lambda x: x[0])
        expirieslist = Remove(np.array(expiries))
        '''
        moneyness = filteredBatchCoordinates.applymap(lambda x: x[1]) if type(filteredBatchCoordinates) == type(pd.DataFrame()) else filteredBatchCoordinates.map(lambda x: x[1])
        moneynesslist = Remove(moneyness)
        '''
        MoneynessRanklist = Remove(MoneynessRanks)
        for ttm in expirieslist:
            vols = []
            calib_vols = []
            strikes = []
            Neural_Net = []

            for i, t in enumerate(filteredBatchCoordinates):
                if t[0] == ttm:
                    try:
                        calib_vols.append(self.model_vol[ref_date, t[0], t[1]])
                        strikes.append(t[1])
                        vols.append(filteredBatch.loc[filteredBatchCoordinates.index[i]])
                        if outputs_val is not None:
                            Neural_Net.append(NeuralNet_val.loc[filteredBatchCoordinates.index[i]])
                    except KeyError:
                        continue
            mse = np.mean(np.square(np.array(vols) - np.array(calib_vols))) ** 0.5

            # Create parallel shift
            #if round(ttm,1) != 3.8:
            #    mse = 0

            if flag_shift == True:
                self.model_shift[ref_date, ttm] = mse
            else:
                self.model_shift[ref_date, ttm] = 0

            h = self.model_shift[ref_date, ttm] * np.ones(len(vols))

            plt.plot(strikes, vols, label="Market")
            plt.plot(strikes, calib_vols - h, label="Heston")

            if outputs_val is not None:
                plt.plot(strikes, Neural_Net, label="Neural Net")

            plt.xlabel('Log Moneyness', fontsize=8)
            plt.ylabel('Volatility', fontsize=8)
            # plt.ylim([0.25, 0.7])
            plt.title('Maturity: ' + str(round(ttm, 1)), fontsize=10)
            plt.legend(fontsize=8)
            plt.show()

        plottingTools.printDelimiter()

        # 2 - plot vol per moneyness
        for m in MoneynessRanklist:
            vols = []
            calib_vols = []
            ttm = []
            Neural_Net = []

            for i, t in enumerate(filteredBatchCoordinates):
                # Get Rank of t[1]

                try:
                    MnRank = MoneynessRanks[filteredBatch.index[i]]
                except KeyError:
                    continue

                if MnRank == m:
                    try:
                        h = self.model_shift[ref_date, t[0]]
                        calib_vols.append(self.model_vol[ref_date, t[0], t[1]] - h)
                        ttm.append(t[0])
                        vols.append(filteredBatch.loc[filteredBatchCoordinates.index[i]])
                        if outputs_val is not None:
                            Neural_Net.append(NeuralNet_val.loc[filteredBatchCoordinates.index[i]])
                    except KeyError:
                        continue

            mse = np.mean(np.square(np.array(vols) - np.array(calib_vols))) ** 0.5
            plt.plot(ttm, vols, label="Market")
            plt.plot(ttm, calib_vols, label="Heston")
            if outputs_val is not None:
                plt.plot(ttm, Neural_Net, label="Neural Net")

            plt.xlabel('Maturity', fontsize=8)
            plt.ylabel('Volatility', fontsize=8)
            # plt.ylim([0.25, 0.7])
            plt.title('Log Moneyness Rank: ' + str(m), fontsize=10)
            plt.legend(fontsize=8)
            plt.show()

        return
