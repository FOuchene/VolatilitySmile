import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import HTML, Image, display
from scipy.optimize import minimize, LinearConstraint, linprog, NonlinearConstraint
import pickle
from Code import interpolator, stochasticModelQL, loadData, plottingTools
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale


def lossReconstruction(s1,s2):#One day
    return np.nanmean((s1-s2)**2)**(0.5)
    #return np.mean(np.mean(((s1-s2))**2, axis=1)**(0.5))


def errorPlotSmile(rmse, relative_rmse, dataList, reconstructed_surface, MaskCoordinates, worstDayRef, rnkMat):
    # dataList refers to complete orginal data

    cells = [[rmse.mean(), relative_rmse.mean()],
             [rmse.max(), relative_rmse.max()],
             [rmse.idxmax().strftime('%Y-%m-%d'),relative_rmse.idxmax().strftime('%Y-%m-%d')]]
    columns = ["Interpolation-Absolute RMSE error", "Interpolation-Relative RMSE error"]
    index = ["Daily average error", "Worst dataset error", "Worst dataset day"]
    print("")
    display(HTML(pd.DataFrame(cells, index=index, columns=columns).to_html()))
    print("")

    # Plot smile per maturity
    coordinates = dataList[1]
    batch = dataList[0]
    reconstructed_batch = reconstructed_surface.loc[worstDayRef]
    Masq = MaskCoordinates.loc[worstDayRef]

    expiries = coordinates.applymap(lambda x: x[0]) if type(coordinates) == type(pd.DataFrame()) else coordinates.map(lambda x: x[0])
    expirieslist = stochasticModelQL.Remove(expiries)
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
                    calib_vols.append(reconstructed_batch.loc[coordinates.index[i]])
                    strikes.append(t[1])
                    market_vols.append(batch.loc[coordinates.index[i]])
                    cle = Masq.loc[coordinates.index[i]]
                    if not cle :
                        interp_vols.append(reconstructed_batch.loc[coordinates.index[i]])
                    else:
                        interp_vols.append(None)
                except KeyError:
                    continue

        mse = np.mean(np.square(np.array(market_vols) - np.array(calib_vols))) ** 0.5
        plt.plot(strikes, market_vols, label="Market")
        plt.plot(strikes, calib_vols, label="Neural Net")
        if rnk in rnkMat:
            plt.plot(strikes, interp_vols, marker='x', label="Out of Sample")

        plt.xlabel('Log Moneyness', fontsize=8)
        plt.ylabel('Volatility', fontsize=8)
        plt.title('Maturity: ' + str(round(ttm, 1)) + 'Y' + ' || Mean Square Error: ' + str(round(100 * mse, 1)) + '%',
                  fontsize=10)
        plt.legend(fontsize=8)
        plt.show()


class Teacher :
    def __init__(self,
                 model, 
                 dataSet,
                 nbEpochs,
                 nbStepCalibrations):
        self.model = model
        self.dataSet = dataSet
        self.nbEpochs = nbEpochs
        self.nbStepCalibrations = nbStepCalibrations
        self.saveResults = True
        
        #Hard-coded parameters for expert user
        self.diagnoseOriginalData = False
        self.colorMapSystem = None
        self.plotType = None
        
        #Temporary results in cache
        self.testingLoss = None
        self.outputs_val = None
        self.codings_val = None
        self.training_val = None
        self.codings_Train = None
        
    #Fit hold model no training data
    def fit(self, InputTrain = None, restoreResults = False):
        if restoreResults : 
            self.testingLoss = self.readObject("trainingLoss")
        else :
            if InputTrain == None:
                dataList = self.dataSet.getTrainingDataForModel()
            else:
                dataList = self.dataSet.getTrainingDataForModel()
            self.testingLoss = self.model.train(dataList, self.nbEpochs, self.dataSet.getTestingDataForModel())


        if self.saveResults :
            self.serializeObject(self.testingLoss, "trainingLoss")
        return
    
    def serializeObject(self, object, objectName):
        fileName = self.model.metaModelName.replace(".cpkt","") + objectName
        #Delete former file version
        #for f in glob.glob(fileName + "*"):
        #    os.remove(f)
        
        with open(fileName, "wb") as f :
            pickle.dump(object, f, protocol=3)
        
        return
    
    def readObject(self, objectName):
        fileName = self.model.metaModelName.replace(".cpkt","") + objectName
        with open(fileName, "rb") as f :
            object = pickle.load(f)
        
        return object
    
    def saveModel(self, session, pathFile): 
        cpkt = self.saver.save(session, pathFile, 
                               latest_filename="LatestSave")
        return cpkt
    #Evaluate the model for given list of dates
    def evalModel(self, dates):
        _, outputs, factors = self.model.evalModel(self.dataSet.getDataForModel(dates))
        return self.dataSet.formatModelDataAsDataSet(outputs), factors
    
    
    
    #Two behaviours : comparison of allocation, comparison of vega strategy
    #Comparison of allocation : Compute the cumulative sum of discounted completion loss weighted by vega exposure (VegaAllocation)
    #Comparison of vega strategy : Compute the cumulative sum of discounted implied volatility moves weighted by one the two vega strategy
    #Here vega exposure means of much move our portfolio when implied volatility surface moves
    #Another metric normalized P&L explains how much percentage of the portfolio variation can be explained with the completed surface
    #Compute the sum of discounted completion loss weighted by vega exposure (VegaAllocation)
    #Here vega exposure means of much move our portfolio when implied volatility surface moves
    #Another metric normalized P&L explains how much percentage of the portfolio variation can be explained with the completed surface
    def ProfitAndLossVegaStrategies(self,
                                    trueDataSerie, 
                                    testedVegaAllocation,
                                    originalVegaAllocation, 
                                    riskFreeRate = 0):
        #Cumulative sum of daily P&L 
        #where daily P&L is the inner product of volatility variation and benchmark vega allocation
        PAndLOriginalVega = 0 
        #Cumulative sum of daily P&L 
        #where daily P&L is the inner product of volatility variation and projected vega allocation
        PAndLTestedVega = 0 
        #Gap between PAndLOriginalVega and PAndLTestedVega
        PAndLTrackingCost = 0 
        PAndLTrackingCostL2 = 0 
        
        PAndLOriginalVegaHistory = []
        PAndLTestedVegaHistory = []
        PAndLTrackingCostHistory = []
        PAndLTrackingCostL2History = []
        dailyTrackingCostHistory = []
        investablePoints = self.dataSet.decideInvestableInstruments()
        
        #If no benchmark vega allocation is provided then we assume a uniform exposure normalized 
        #such that sum of vega is equal to one
        
        #trueVariation  = (trueDataSerie[0].diff()).dropna(how="all", axis=0)
        
        #Determine previous day as the previous business day (ignoring holiday for now)
        previousDay = trueDataSerie[0].index[0]
        #previousDay += pd.tseries.offsets.BusinessDay(n = -1)
        
        for day, gapSurface in trueDataSerie[0].tail(-1).iterrows(): #for each day
            #Cumulative sum of relative signed gap weighted by vega exposure
            deltaDay = day - previousDay
            discountStep = (riskFreeRate * float(deltaDay.days) / 365.0)
            
            dailyTrueVariation = (trueDataSerie[0].loc[day][investablePoints[previousDay][1]].values - 
                                  trueDataSerie[0].loc[previousDay][investablePoints[previousDay][0]].values)
            dailyTrueVariationSerie = pd.Series(dailyTrueVariation, 
                                                index = investablePoints[previousDay][0])
            #dailyPAndLOriginalVega = np.nansum(gapSurface * originalVegaAllocation.loc[day])
            dailyPAndLOriginalVega = np.nanmean(dailyTrueVariation)
            PAndLOriginalVega += PAndLOriginalVega * discountStep - dailyPAndLOriginalVega
            PAndLOriginalVegaHistory.append(PAndLOriginalVega)
            
            dailyAllocation = testedVegaAllocation.loc[day]
            dailyPAndLTestedVega = np.nansum(dailyTrueVariationSerie[dailyAllocation.index] * dailyAllocation) 
            PAndLTestedVega += PAndLTestedVega * discountStep - dailyPAndLTestedVega
            PAndLTestedVegaHistory.append(PAndLTestedVega)
            
            dailyTrackingCost = dailyPAndLTestedVega - dailyPAndLOriginalVega 
            PAndLTrackingCost += PAndLTrackingCost * discountStep - dailyTrackingCost
            PAndLTrackingCostHistory.append(PAndLTrackingCost)
            dailyTrackingCostHistory.append(dailyTrackingCost)
            
            dailyTrackingCostL2 = np.square(dailyPAndLTestedVega - dailyPAndLOriginalVega)
            PAndLTrackingCostL2 = np.sqrt( np.square(PAndLTrackingCostL2) + np.square(PAndLTrackingCostL2 * discountStep) + dailyTrackingCostL2)
            PAndLTrackingCostL2History.append(PAndLTrackingCostL2)
            
            previousDay = day
            
        
        originalVegaPAndLSerie = pd.Series(PAndLOriginalVegaHistory, index=trueDataSerie[0].tail(-1).index).rename("P&L Vanilla Vega")
        testedVegaPAndLSerie = pd.Series(PAndLTestedVegaHistory, index=trueDataSerie[0].tail(-1).index).rename("P&L Projected Vega")
        trackingCostPAndLSerie = pd.Series(PAndLTrackingCostHistory, index=trueDataSerie[0].tail(-1).index).rename("Tracking cost")
        trackingCostL2PAndLSerie = pd.Series(PAndLTrackingCostL2History, index=trueDataSerie[0].tail(-1).index).rename("Tracking L2 cost")
        dailyTrackingCostSerie = pd.Series(dailyTrackingCostHistory, index=trueDataSerie[0].tail(-1).index).rename("Daily error")
        
        nbHedgingPeriods = trueDataSerie[0].tail(-1).shape[0] 
        index=["P&L variation divided by True P&L variation",
               "Percentage of cumulative P&L variation explained", 
               "Tracking cost of completion portfolio", 
               "Tracking cost L2 of completion portfolio",
               "Tracking cost of completion portfolio divided by total variation of true portfolio"]
        columns = ["Daily", "Annualized (255 days a year)", "Total"]
        
        PAndLQuotient = PAndLTestedVega / PAndLOriginalVega
        RelativeTrackingCost = PAndLTrackingCost/PAndLOriginalVega
        cumulativePAndExplained = abs((originalVegaPAndLSerie.cumprod() / testedVegaPAndLSerie.cumprod()).iloc[-1])
        cells = [[PAndLQuotient,PAndLQuotient,PAndLQuotient],
                 [cumulativePAndExplained**(1.0 / nbHedgingPeriods),
                  cumulativePAndExplained**(255.0 / nbHedgingPeriods),
                  cumulativePAndExplained],
                 [PAndLTrackingCost*(1.0 / nbHedgingPeriods),
                  PAndLTrackingCost*(255.0 / nbHedgingPeriods),
                  PAndLTrackingCost],
                 [PAndLTrackingCostL2*np.sqrt(1.0 / nbHedgingPeriods), #standard deviation
                  PAndLTrackingCostL2*np.sqrt(255.0 / nbHedgingPeriods), 
                  PAndLTrackingCostL2],
                 [RelativeTrackingCost,RelativeTrackingCost,RelativeTrackingCost]]
        
        summary = pd.DataFrame(cells, index=index, columns=columns)
        print()
        display(HTML(summary.to_html()))
        print()
        
        plottingTools.plotSeries([originalVegaPAndLSerie, testedVegaPAndLSerie, trackingCostPAndLSerie],
                                 title="P&L performance")
        plottingTools.plotSeries([originalVegaPAndLSerie, testedVegaPAndLSerie, trackingCostL2PAndLSerie],
                                 title="P&L performance")
        if dailyTrackingCostSerie.std() > 1e-6 :
            dailyTrackingCostSerie.plot.kde(bw_method=0.5)
        refSize=5
        plt.ylabel("Density", fontsize=2*refSize, labelpad=3*refSize)
        plt.xlabel("Tracking error", fontsize=2*refSize, labelpad=3*refSize)
        plt.show()
        return summary
    
        
    #Compute the sum of discounted completion loss weighted by vega exposure (VegaAllocation)
    #Here vega exposure means of much move our portfolio when implied volatility surface moves
    #Another metric normalized P&L explains how much percentage of the portfolio variation can be explained with the completed surface
    def ProfitAndLoss(self,
                      trueDataSerie, 
                      ApproximatedDataSerie, 
                      VegaAllocation = None, 
                      riskFreeRate = 0):
        totalVariationAccount = 0 #Cost of tracking real portfolio with a cash position
        completionVariationAccount = 0 #Cost of tracking completion portfolio with a cash position
        PAndLCompletion = 0 #Tracking cost between completion portfolio and real portfolio
        trackingErrorL2 = 0
        PAndL = 0 #
        cumulativeAccountExplained = 1
        
        totalVariationAccountHistory = []
        PAndLCompletionHistory = []
        TrackingErrorL2History = []
        dailyLossSerie = []
        
        
        
        #If no vega allocation is provided then we assume a uniform exposure normalized 
        #such that sum of vega is equal to one
        nbPoints = trueDataSerie.shape[1]
        usedVegaAllocation = VegaAllocation if VegaAllocation else (np.ones(nbPoints)/float(nbPoints))
        
        approximationCompletionError = ApproximatedDataSerie - trueDataSerie
        #approximatedVariation = (ApproximatedDataSerie - trueDataSerie.shift()).dropna(how="all")
        #completionVariation  = (ApproximatedDataSerie.diff()).dropna(how="all")
        #trueVariation  = (trueDataSerie.diff()).dropna(how="all")
        
        investablePoints = self.dataSet.decideInvestableInstruments()
        
        #Determine previous day as the previous business day (ignoring holiday for now)
        previousDay = trueDataSerie.index[0]
        #previousDay += pd.tseries.offsets.BusinessDay(n = -1)
        
        for day, gapSurface in approximationCompletionError.iterrows(): #for each day
            #Cumulative sum of relative signed gap weighted by vega exposure
            deltaDay = day - previousDay
            
            if day != approximationCompletionError.index[0] :
                #dailyLoss = np.sum(gapSurface * usedVegaAllocation) 
                dailyCompletionError = gapSurface[investablePoints[previousDay][1]]
                dailyLoss =  np.nanmean(dailyCompletionError)
                PAndL += PAndL * (riskFreeRate * float(deltaDay.days) / 365.0) - dailyLoss
                
                #dailyTrueVariation = np.nansum(trueVariation.loc[day] * usedVegaAllocation)
                dailyDiff = (trueDataSerie.loc[day][investablePoints[previousDay][1]].values - 
                             trueDataSerie.loc[previousDay][investablePoints[previousDay][0]].values)
                dailyTrueVariation = np.nanmean(dailyDiff)
                
                #dailyLoss = np.nansum(approximationCompletionError.loc[day] * usedVegaAllocation)
                
                dailyApproxDiff = (ApproximatedDataSerie.loc[day][investablePoints[previousDay][1]].values - 
                                   trueDataSerie.loc[previousDay][investablePoints[previousDay][0]].values)
                dailyApproxVariation = np.nanmean(dailyApproxDiff)
                
                dailyCompletionDiff = (ApproximatedDataSerie.loc[day][investablePoints[previousDay][1]].values - 
                                       ApproximatedDataSerie.loc[previousDay][investablePoints[previousDay][0]].values)
                dailyCompletionVariation = np.nanmean(dailyCompletionDiff)
                
                discountStep = (riskFreeRate * float(deltaDay.days) / 365.0)
                
                totalVariationAccount += totalVariationAccount * discountStep - dailyTrueVariation
                completionVariationAccount += completionVariationAccount * discountStep - dailyCompletionVariation
                PAndLCompletion += PAndLCompletion * discountStep - dailyLoss
                trackingErrorL2 = np.sqrt(np.square(trackingErrorL2) + np.square(trackingErrorL2 * discountStep) + np.square(dailyLoss))
                
                PAndLCompletionHistory.append(PAndLCompletion)
                dailyLossSerie.append(dailyLoss)
                totalVariationAccountHistory.append(totalVariationAccount)
                TrackingErrorL2History.append(trackingErrorL2)
                
                cumulativeAccountExplained *= abs(dailyApproxVariation/dailyTrueVariation)
                
            else :
                PAndL = 0
            
            previousDay = day
            
        
        
        
        nbHedgingPeriods = approximationCompletionError.shape[0] - 1
        index=["Completion P&L variation divided by True P&L variation",
               "Percentage of cumulative P&L variation explained", 
               "Tracking cost of completion portfolio",
               "Tracking L2 cost of completion portfolio",
               "Tracking cost of completion portfolio divided by total variation of true portfolio"]
        columns = ["Daily", "Annualized (255 days a year)", "Total"]
        
        PAndLQuotient = completionVariationAccount / totalVariationAccount
        RelativeTrackingCost = PAndLCompletion/totalVariationAccount
        cells = [[PAndLQuotient,PAndLQuotient,PAndLQuotient],
                 [cumulativeAccountExplained**(1.0 / nbHedgingPeriods),
                  cumulativeAccountExplained**(255.0 / nbHedgingPeriods),
                  cumulativeAccountExplained],
                 [PAndLCompletion*(1.0 / nbHedgingPeriods),
                  PAndLCompletion*(255.0 / nbHedgingPeriods),
                  PAndLCompletion],
                 [trackingErrorL2 * np.sqrt(1.0 / nbHedgingPeriods), #standard deviation
                  trackingErrorL2 * np.sqrt(255.0 / nbHedgingPeriods), 
                  trackingErrorL2],
                 [RelativeTrackingCost,RelativeTrackingCost,RelativeTrackingCost]]
        
        summary = pd.DataFrame(cells, index=index, columns=columns)
        print()
        display(HTML(summary.to_html()))
        print()
        
        plottingTools.plotSeries([pd.Series(PAndLCompletionHistory, index=trueDataSerie.tail(-1).index).rename("Tracking cost"),
                                  pd.Series(totalVariationAccountHistory, index=trueDataSerie.tail(-1).index).rename("Total Variation")],
                                 title="P&L performance")
        plottingTools.plotSeries([pd.Series(TrackingErrorL2History, index=trueDataSerie.tail(-1).index).rename("Tracking L2 cost"),
                                  pd.Series(totalVariationAccountHistory, index=trueDataSerie.tail(-1).index).rename("Total Variation")],
                                 title="P&L performance")
        #Plot loss density
        
        if np.std(dailyLossSerie) > 1e-6 :
            pd.Series(dailyLossSerie, index=trueDataSerie.tail(-1).index).plot.kde(bw_method=0.5)
        refSize = 5
        plt.ylabel("Density", fontsize=2*refSize, labelpad=3*refSize)
        plt.xlabel("Tracking error", fontsize=2*refSize, labelpad=3*refSize)
        plt.show()
        return summary

    # Intérêt de la fonction  ci-dessous à préciser??
    # Devrait plutôt check entre input ACP explained variance en fonction du nombre de latent factor
    def latentfactorSanityCheck(self) : 
        encodingCorrelation = self.codings_Train.corr()
        print(encodingCorrelation)
        if encodingCorrelation.dropna().size > 0 :
            pca = PCA(n_components=encodingCorrelation.shape[0])
            _ = pca.fit_transform(scale(self.codings_Train.corr()))
            plt.plot(pca.explained_variance_ratio_)
            plt.title("Eigen value for latent space")
            plt.show()
    
    def printThetaArbitrage(self, historyPred, historyRef, codings):
        
        plottingTools.printDelimiter()
        print("Calendar arbitrage")
        errorsAbsRMSE = pd.Series(np.nanmean(np.square(historyRef - historyPred),axis=1)**0.5, 
                              index = historyRef.index)
        worstDayPred, worstDayRef = plottingTools.getWorstGrids(historyPred,
                                                                historyRef,
                                                                errorsAbsRMSE)
        modelData = self.dataSet.getDataForModel(worstDayRef.name)
        coordinates = self.dataSet.formatModelDataAsDataSet(modelData)[1]
        encodings = codings.loc[worstDayRef.name]
        thetaSurface = self.model.getArbitrageTheta(modelData, encodings)
        plottingTools.plotGrid(thetaSurface.iloc[0],
                               coordinates,    
                               "Calendar condition for worst reconstruction on testing dataset",
                               colorMapSystem=self.colorMapSystem,
                               plotType=self.plotType,
                               refPoints = None,
                               zLabelUser = "Implied total variance Theta")
        
        print("Minimal theta : ", thetaSurface.min().min())
        
        plottingTools.printDelimiter()
        return
    
    #Plot some results for compression
    def diagnoseCompression(self, TestingdataList = None, TrainingdataList = None, restoreResults = False):
        if self.testingLoss is None :
            plottingTools.printIsolated("Please fit model on data before any diagnosis")
            return
        
        if restoreResults :
            resCompression = self.readObject("compressionResult")
            self.outputs_val = resCompression["outputs_val"]
            self.codings_val = resCompression["codings_val"]
            self.training_val = resCompression["training_val"]
            self.codings_Train = resCompression["codings_Train"]
        else :
            if TestingdataList == None:
                _, self.outputs_val, self.codings_val = self.model.evalModel(self.dataSet.getTestingDataForModel())
                _, self.training_val, self.codings_Train = self.model.evalModel(self.dataSet.getTrainingDataForModel())
            else:
                _, self.outputs_val, self.codings_val = self.model.evalModel(TestingdataList)
                _, self.training_val, self.codings_Train = self.model.evalModel(TrainingdataList)


        if (TestingdataList == None) or (TrainingdataList == None):
            refTestingValues = self.dataSet.formatModelDataAsDataSet(self.dataSet.getTestingDataForModel())[0]
            refTrainingValues = self.dataSet.formatModelDataAsDataSet(self.dataSet.getTrainingDataForModel())[0]
        else:
            refTestingValues = self.dataSet.formatModelDataAsDataSet(TestingdataList)[0]
            refTrainingValues = self.dataSet.formatModelDataAsDataSet(TrainingdataList)[0]

        predTestingValues = self.dataSet.formatModelDataAsDataSet(self.outputs_val)
        predTrainingValues = self.dataSet.formatModelDataAsDataSet(self.training_val)
        
        if self.saveResults :
            resCompression = {}
            resCompression["outputs_val"] = self.outputs_val
            resCompression["codings_val"] = self.codings_val
            resCompression["training_val"] = self.training_val
            resCompression["codings_Train"] = self.codings_Train
            resCompression["predTestingValues"] = predTestingValues
            resCompression["refTestingValues"] = refTestingValues
            resCompression["predTrainingValues"] = predTrainingValues
            resCompression["refTrainingValues"] = refTrainingValues
            self.serializeObject(resCompression, "compressionResult")
        
        plottingTools.diagnoseModels(self.codings_val,
                                     predTrainingValues,
                                     refTrainingValues,
                                     predTestingValues,
                                     refTestingValues,
                                     self.testingLoss,
                                     self.dataSet,
                                     colorMapSystem=self.colorMapSystem,
                                     plotType=self.plotType)
        
        plottingTools.printDelimiter()
        
        self.printThetaArbitrage(predTestingValues, refTestingValues, self.codings_val)
        
        plottingTools.printDelimiter()
        
        self.latentfactorSanityCheck()
        plottingTools.printDelimiter()

        return
    
    #Complete surface for a given date
    def completionTest(self, date):
        if self.outputs_val is None :
            raise ValueError("Diagnose compression before completing one day")
        
        fullDataSet = self.dataSet.getDataForModel(date)
        factorHistory = self.codings_Train.append(self.codings_val)
        deletedIndex = self.dataSet.maskedPoints
        
        #Delete points inside the surface
        surfaceToComplete = fullDataSet[0]
        surfaceSparse = self.dataSet.maskDataset(surfaceToComplete)
        
        #Get latest available values for latent variables
        #lastFactorsValues = factorHistory[ factorHistory.index < date].iloc[-1]
        lastFactorsValues = self.selectClosestObservationsInThePast(date,
                                                                    factorHistory,
                                                                    surfaceSparse)
        
        #Complete the surface
        l, f, s, lSerie = self.executeCompletion([surfaceSparse] + fullDataSet[1:], 
                                                 lastFactorsValues, 
                                                 self.nbStepCalibrations)

        plottingTools.plotLossThroughEpochs(lSerie,
                                            title = "Calibration loss on non-missing points through epochs")
        
        originalSurface = self.dataSet.formatModelDataAsDataSet(surfaceToComplete)
        outputSurface = pd.Series(self.dataSet.formatModelDataAsDataSet(s), index = surfaceToComplete.index)
        
        plottingTools.printIsolated("L2 Reconstruction loss : ",
                                    lossReconstruction(originalSurface,outputSurface))
        return l, f, outputSurface, originalSurface
        
    #Show surface sensitivity with respect to each factor
    def printOutputSensiToFactors(self, factorCalibrated, date):
        
        allData = self.dataSet.getDataForModel(date)
        s, JFactors = self.model.evalSingleDayWithoutCalibrationWithSensi(factorCalibrated, allData)
        sIndexed = pd.Series(s,index = allData[0].index)
        JFactorsDf = pd.DataFrame(JFactors, index = allData[0].index)
        
        title = "Original Data"
        originalCoordinates = self.dataSet.formatModelDataAsDataSet(allData)[1]
        plottingTools.plotGrid(sIndexed, originalCoordinates, title,
                               colorMapSystem=self.colorMapSystem,
                               plotType=self.plotType)
        
        plottingTools.printDelimiter()
        plottingTools.printIsolated("Sensitivities to each factor")
        for k in JFactorsDf.columns:
            title = "Data Sensitivity to factor number " + str(k)
            plottingTools.plotGrid(JFactorsDf[k], originalCoordinates, title,
                                   colorMapSystem=self.colorMapSystem,
                                   plotType=self.plotType)
        return
    
    def correctExtrapolationDomain(self, sparseSurface, completedSurface, coordinates):
        extrapolationMode = (self.model.hyperParameters["extrapolationMode"] 
                             if "extrapolationMode" in self.model.hyperParameters else "NoExtrapolation")
        #remove values from extrapolation domain 
        interpolatedPoint = None
        if extrapolationMode == 'InnerDomain' :
            interpolatedPoint = interpolator.areInInnerPolygon(sparseSurface, coordinates)
        elif extrapolationMode == 'OuterDomain' :
            interpolatedPoint = interpolator.areInOuterPolygon(sparseSurface, coordinates)
        else : #NoExtrapolation, keep all points
            interpolatedPoint = coordinates.loc[sparseSurface.index]
        
        #Hide data not in the domain
        pointTokeep = interpolatedPoint.index
        pointToRemove = sparseSurface.index.difference(pointTokeep)
        completedSurface.loc[pointToRemove] = np.NaN
        
        extrapolatedSurface = interpolator.extrapolationFlat(completedSurface, coordinates)
        
        return extrapolatedSurface.rename(sparseSurface.name)
        
    
    #Intermediate step for possibly using flat extrapolation
    def executeCompletion(self, 
                          sparseSurfaceList, 
                          initialValueForFactors, 
                          nbCalibrationStep):
        
        l, f, S, lSerie = self.model.completeDataTensor(sparseSurfaceList,
                                                        initialValueForFactors, 
                                                        nbCalibrationStep)
        #Retain only completed points
        completedPoints = S[sparseSurfaceList[0].isna()].append(sparseSurfaceList[0].dropna()).rename(S.name)[S.index]
        
        
        extrapolatedS = self.correctExtrapolationDomain(sparseSurfaceList[0], completedPoints, sparseSurfaceList[1])
        
        return l, f, extrapolatedS, lSerie 
    
    def selectClosestObservationsInThePast(self, dayObs, factorHistory, incompleteSurface):
        wholeHistory = self.dataSet.getDataForModel()[0]
        #Get all previous observations
        history = wholeHistory[wholeHistory.index < dayObs]
        error = np.square(history - incompleteSurface).dropna(axis=1)
        argMinError = error.mean(axis=1).idxmin()
        return factorHistory.loc[argMinError]    

    def selectPreviousObservationsInThePast(self, dayObs, factorHistory):
        wholeHistory = self.dataSet.getDataForModel()[0]
        #Get all previous observations
        history = wholeHistory[wholeHistory.index < dayObs]
        argMinError = history.tail(1).index
        return factorHistory.loc[argMinError].transpose()

    # Assess completion along testing data history and recalibrate from latest available factor values
    def diagnoseCompletion(self, dataList, rnkMat, rnkMn, restoreResults=False):
        #if self.outputs_val is None:
        #    raise ValueError("Diagnose compression before completing one day")

        if restoreResults:
            result = self.readObject("completion")
        else:
            filteredData = loadData.FilterOnMaturityMoneneynessRank(dataList[0], dataList[1], dataList[2], rnkMat, rnkMn)
            sparse_dataList = [filteredData[0], filteredData[1], filteredData[2]]
            listMoneynessRank, listMaturityRank, MaskCoordinates = filteredData[3], filteredData[4], filteredData[5]
            complete_dataList = loadData.removePointsWithInvalidCoordinatesWithFwd(dataList[0], dataList[1], dataList[2])


            # Evaluate factors on the testingSet to get real factors
            _, trueSurface, trueFactors = self.model.evalModel(complete_dataList)
            trueSurface = self.dataSet.formatModelDataAsDataSet(complete_dataList[0])
            _, completedSurfaces, _, _ = self.model.calibratedFactors2(complete_dataList, trueFactors)
            testingSet = complete_dataList

            calibrationLosses = pd.DataFrame(completedSurfaces-trueSurface, index=testingSet[0].index)
            calibrationRelativeLosses = pd.DataFrame((completedSurfaces - trueSurface)/trueSurface, index=testingSet[0].index)
            Surfaces = pd.DataFrame(completedSurfaces, index=testingSet[0].index, columns=testingSet[0].columns)

            rmse = []
            l1 = []
            relative_rmse = []
            relative_l1 = []
            rmse_interp = []
            relative_rmse_interp = []

            # Computing errors and plotting results
            for i in range(calibrationLosses.shape[0]):
                Losses = calibrationLosses.iloc[i]
                RelativeLosses = calibrationRelativeLosses.iloc[i]
                Masq = MaskCoordinates.iloc[i]
                err_interp = [x for j, x in enumerate(Losses) if Masq[Losses.index[j]]==False]

                relerr_interp = [x for j, x in enumerate(RelativeLosses) if Masq[Losses.index[j]]==False]
                err = Losses
                relerr = RelativeLosses

                rmse.append(np.nanmean(np.square(err)) ** 0.5)
                l1.append(np.nanmean(np.absolute(err)))
                relative_rmse.append(np.nanmean(np.square(relerr)) ** 0.5)
                relative_l1.append(np.nanmean(np.absolute(relerr)))

                rmse_interp.append(np.nanmean(np.square(err_interp)) ** 0.5)
                relative_rmse_interp.append(np.nanmean(np.square(relerr_interp)) ** 0.5)

            self.rmse = pd.Series(rmse, index=testingSet[0].index)
            self.l1 = pd.Series(l1, index=testingSet[0].index)
            self.relative_rmse = pd.Series(relative_rmse, index=testingSet[0].index)
            self.relative_l1 = pd.Series(relative_l1, index=testingSet[0].index)

            self.rmse_interp = pd.Series(rmse_interp, index=testingSet[0].index)
            self.relative_rmse_interp = pd.Series(relative_rmse_interp, index=testingSet[0].index)

            stochasticModelQL.errorPlot(self.rmse, self.l1, self.relative_rmse, self.relative_l1)

            # Per maturity graphs of worst day prediction
            worstDayRef = self.rmse.idxmax()
            batch = testingSet[0].loc[worstDayRef]
            batchcoordinates = testingSet[1].loc[worstDayRef]
            filteredBatch, filteredBatchCoordinates = loadData.removePointsWithInvalidCoordinates(batch, batchcoordinates)
            dataList = [filteredBatch, filteredBatchCoordinates]
            
            errorPlotSmile(self.rmse_interp, self.relative_rmse_interp, dataList, Surfaces, MaskCoordinates, worstDayRef, rnkMat)

        return None

    def plotBackTestCompletion(self, result):

        print("Hello World")

        return
        #

