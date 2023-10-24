import os
import pandas as pd
import numpy as np
from scipy.optimize import minimize, LinearConstraint
import time
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pickle
import stochasticModelQL

# Define Column Name
indexName = 'date'
indexExpiry = 'optionExpiry'
indexTenor = 'underlyingTerm'
indexStrike = 'Strike'
indexRelStrike = 'RelativeStrike'


def getTTMFromCoordinates(dfList):
    return dfList[1].applymap(lambda x: x[0])


def getMoneynessFromCoordinates(dfList):
    return dfList[1].applymap(lambda x: x[1])


def parseTerm(stringTerm):
    if 'M' == stringTerm[-1]:
        return float(stringTerm[:-1]) / 12
    elif 'Y' == stringTerm[-1]:
        return float(stringTerm[:-1])
    else:
        raise Exception("Can not parse term")


def parseTenor(row):
    return [parseTerm(row['underlyingTerm']), parseTerm(row['optionExpiry'])]


def smileFromSkew(skew):
    atmVol = skew['A']
    # smile = atmVol + skew[skewShift]
    # return smile#.append(skew.drop(smile.index))
    return atmVol + skew.drop('A')


def parseStrike(relStrike):
    if relStrike.name[3] == 'A':
        return relStrike['forward']
    if "+" in relStrike.name[3]:
        shift = int(relStrike.name[3].split("+")[1])
        return relStrike['forward'] + shift / 1000
    if "-" in relStrike.name[3]:
        shift = int(relStrike.name[3].split("-")[1])
        return relStrike['forward'] - shift / 1000
    raise Exception(' Can not parse Strike ')


def equalDf(df1, df2):
    if df1.shape == df2.shape:
        if np.sum(np.isnan(df1.values)) != np.sum(np.isnan(df2.values)):
            print("Not the same number of nan")
            return False
        tol = 1e-6
        gap = np.nansum(np.abs(df1.values - df2.values))
        if gap < tol:
            return True
        else:
            print("Large df error : ", gap)
            return False
    print("Not the same shape")
    return False


# Encapsulation class for Sklearn Standard scaling
class customMeanStdScale:
    def __init__(self, feature_range=(0, 1)):
        self.scalerList = []

    # We can enforce the minimum if we expect smaller data in the testing set
    def fit(self, dataset,
            enforceDataSetMin=None,
            enforceDataSetMax=None):
        hasTupleElt = (type(dataset.iloc[0, 0] if dataset.ndim == 2 else dataset.iloc[0]) == type(tuple()))
        if hasTupleElt:
            tupleSize = len(dataset.iloc[0, 0] if dataset.ndim == 2 else dataset.iloc[0])
            self.scalerList = [StandardScaler() for i in range(tupleSize)]
            for k in range(tupleSize):
                funcAccess = lambda x: x[k]
                scaler = self.scalerList[k]
                dfElt = dataset.applymap(funcAccess) if (type(dataset) != type(pd.Series())) else dataset.map(
                    funcAccess)
                scaler.fit(dfElt)

        else:
            self.scalerList = []
            self.scalerList.append(StandardScaler())
            self.scalerList[0].fit(dataset)
        return

    def transformSingleDf(self, scaler, dfElt):
        totalVariance = np.sum(scaler.var_)
        if totalVariance <= 1e-6:  # Avoid mean scaling for constant data
            return dfElt
        if type(dfElt) == type(pd.Series()):
            return pd.Series(np.ravel(scaler.transform(dfElt.values.reshape(1, -1))),
                             index=dfElt.index).rename(dfElt.name)
        return pd.DataFrame(scaler.transform(dfElt),
                            index=dfElt.index,
                            columns=dfElt.columns)

    def transform(self, dataset):
        hasTupleElt = (type(dataset.iloc[0, 0] if dataset.ndim == 2 else dataset.iloc[0]) == type(tuple()))
        if hasTupleElt:
            tupleSize = len(dataset.iloc[0, 0] if dataset.ndim == 2 else dataset.iloc[0])
            scaledDfList = []
            for k in range(tupleSize):
                funcAccess = lambda x: x[k]
                dfElt = dataset.applymap(funcAccess) if (type(dataset) != type(pd.Series())) else dataset.map(
                    funcAccess)
                scaler = self.scalerList[k]
                scaledDfList.append(np.ravel(self.transformSingleDf(scaler, dfElt).values))
                # Flattened list of tuples
            tupleList = list(zip(*scaledDfList))
            # Merge all datasets into a single structure
            if dataset.ndim == 2:
                reshapedList = [tupleList[(i * dataset.shape[1]):((i + 1) * dataset.shape[1])] for i in
                                range(dataset.shape[0])]
                return pd.DataFrame(reshapedList,
                                    index=dataset.index,
                                    columns=dataset.columns)
            else:
                reshapedList = tupleList
                return pd.Series(reshapedList, index=dataset.index)
        else:
            return self.transformSingleDf(self.scalerList[0], dataset)

        return None

    def inverTransformSingleDf(self, scaler, dfElt):
        totalVariance = np.sum(scaler.var_)
        if totalVariance <= 1e-6:  # Avoid mean scaling for constant data
            return dfElt
        if type(dfElt) == type(pd.Series()):
            return pd.Series(np.ravel(scaler.inverse_transform(dfElt.values.reshape(1, -1))),
                             index=dfElt.index).rename(dfElt.name)
        return pd.DataFrame(scaler.inverse_transform(dfElt),
                            index=dfElt.index,
                            columns=dfElt.columns)

    def inverse_transform(self, scaledDataset):
        hasTupleElt = (type(scaledDataset.iloc[0, 0] if scaledDataset.ndim == 2 else scaledDataset.iloc[0]) == type(
            tuple()))
        if hasTupleElt:
            tupleSize = len(scaledDataset.iloc[0, 0] if scaledDataset.ndim == 2 else scaledDataset.iloc[0])
            scaledDfList = []
            for k in range(tupleSize):
                funcAccess = lambda x: x[k]
                dfElt = scaledDataset.applymap(funcAccess) if (
                        type(scaledDataset) != type(pd.Series())) else scaledDataset.map(funcAccess)
                scaler = self.scalerList[k]
                scaledDfList.append(np.ravel(self.inverTransformSingleDf(scaler, dfElt).values))
                # Flattened list of tuples
            tupleList = list(zip(*scaledDfList))
            # Merge all datasets into a single structure
            if scaledDataset.ndim == 2:
                reshapedList = [tupleList[(i * scaledDataset.shape[1]):((i + 1) * scaledDataset.shape[1])] for i in
                                range(scaledDataset.shape[0])]
                return pd.DataFrame(reshapedList,
                                    index=scaledDataset.index,
                                    columns=scaledDataset.columns)
            else:
                reshapedList = tupleList
                return pd.Series(reshapedList, index=scaledDataset.index)
        else:
            return self.inverTransformSingleDf(self.scalerList[0], scaledDataset)

        return None


def selectLessCorrelatedFeatures(featureCorr, nbPoints):
    objectiveFunction = lambda x: x.T @ featureCorr.values @ x
    gradient = lambda x: (featureCorr.values + featureCorr.values.T) @ x
    hessian = lambda x: featureCorr.values + featureCorr.values.T
    nbRestart = 5
    x0s = np.random.uniform(size=(nbRestart, featureCorr.shape[1]))
    x0s = x0s * nbPoints / np.sum(x0s, axis=1, keepdims=True)
    bestSol = x0s[0, :]
    bestVar = featureCorr.shape[1]

    bounds = [[0, 1]] * featureCorr.shape[1]
    budgetAllocation = LinearConstraint(np.ones((1, featureCorr.shape[1])), [nbPoints], [nbPoints], keep_feasible=True)
    for k in range(nbRestart):
        res = minimize(objectiveFunction, x0s[k, :],
                       bounds=bounds,
                       constraints=budgetAllocation,
                       method="trust-constr",
                       jac=gradient,
                       hess=hessian)
        if (res.fun < bestVar) or (k == 0):
            bestSol = res.x
            bestVar = res.fun
        print("Attempt no ", k, " ; best solution : ", bestSol, " ; best inertia : ", bestVar)

    topnbPointsValue = -(np.sort(-bestSol)[nbPoints - 1])
    optimalAllocation = pd.Series(bestSol, index=featureCorr.index)
    return optimalAllocation[optimalAllocation >= topnbPointsValue].index


# These class are responsible for :
# - passing the right data to the model for trainingData
# - converting data to the original format for plotting 
class datasetATM:
    def __init__(self, pathToDataset,
                 trainingSetPercentage,
                 minExpiry,
                 completionRate,
                 scaleFeatures=False):
        self.trainingSetPercentage = trainingSetPercentage
        self.pathToDataset = pathToDataset
        self.activateScaling = scaleFeatures
        self.isGridStable = True

        self.testVol = None
        self.trainVol = None
        self.VolSerie = None
        self.volScaler = None
        self.scaledTrainVol = None
        self.scaledTestVol = None

        self.testCoordinates = None
        self.trainCoordinates = None
        self.CoordinatesSerie = None
        self.coordinatesScaler = None
        self.scaledTrainCoordinates = None
        self.scaledTestCoordinates = None

        self.testFwd = None
        self.trainFwd = None
        self.FwdSerie = None
        self.fwdScaler = None
        self.scaledTrainFwd = None
        self.scaledTestFwd = None

        self.testStrike = None
        self.trainStrike = None
        self.StrikeSerie = None

        self.loadData()
        self.scaleDataSets()

        lambdaAppend = (lambda x: x[0].append(x[1]) if x[0] is not None else None)
        self.fullHistory = list(map(lambdaAppend, zip(self.getTrainingDataForModel(), self.getTestingDataForModel())))
        self.fullScaler = [self.volScaler, self.coordinatesScaler, self.fwdScaler, None]

        self.gridSize = self.getTestingDataForModel()[0].shape[1]

        return

    def loadData(self):
        raise NotImplementedError("Abstract class")
        return

    def sanityCheck(self):

        print("Testing formatModelDataAsDataSet")
        assert (equalDf(self.testVol.dropna(how="all").head(),
                        self.formatModelDataAsDataSet(self.getTestingDataForModel())[0].head()))

        origData = self.formatModelDataAsDataSet(self.getTrainingDataForModel())

        print("Testing coordinates")
        assert (equalDf(self.trainCoordinates.head().applymap(lambda x: x[0]),
                        origData[1].head().applymap(lambda x: x[0])))
        assert (equalDf(self.trainCoordinates.head().applymap(lambda x: x[1]),
                        origData[1].head().applymap(lambda x: x[1])))

        print("Testing Forward")
        assert (equalDf(self.getTrainingDataForModel()[2].head(),
                        self.convertRealDataToModelFormat(
                            self.formatModelDataAsDataSet(self.getTrainingDataForModel()))[2].head()))

        print("Testing masking function")
        maskedDf = self.maskDataset(self.getTrainingDataForModel()[1]).dropna(how="all", axis=1).head()
        assert (maskedDf.shape[1] == (self.gridSize - self.maskedPoints.size))

        print("Testing convertRealDataToModelFormat")
        assert (equalDf(self.trainVol.loc[origData[0].index].head(),
                        self.formatModelDataAsDataSet(self.convertRealDataToModelFormat(origData))[0].head()))

        print("Success")
        return

    # When the grid is not fixed - i.e. volatilities time to maturities are sliding -
    # we need to decide which instruments can be compared between two dates
    def decideInvestableInstruments(self):
        coordinatesDf = self.formatModelDataAsDataSet(self.getDataForModel())[1]

        pairIndexHistory = []  # series of pair of index
        nextTTMDf = coordinatesDf.shift(-1).dropna(how="all")
        for serie in coordinatesDf.head(-1).iterrows():
            currentDay = serie[1]
            nextDay = nextTTMDf.loc[serie[0]]
            currentRankForHedgeablePoints = currentDay.index
            nextRankForHedgeablePoints = nextDay.index
            pairIndexHistory.append((currentRankForHedgeablePoints, nextRankForHedgeablePoints))
        pairIndexHistory.append((nextRankForHedgeablePoints, nextRankForHedgeablePoints))
        pairIndexHistory = pd.Series(pairIndexHistory, index=coordinatesDf.index)
        return pairIndexHistory

    # List Format : First position vol, second position coordinates, third position forward, fourth position strike
    def getTestingDataForModel(self):
        return [self.scaledTestVol, self.scaledTestCoordinates, self.scaledTestFwd, self.testStrike]

    def getTrainingDataForModel(self):
        return [self.scaledTrainVol, self.scaledTrainCoordinates, self.scaledTrainFwd, self.trainStrike]

    def getDataForModel(self, dates=None):
        if dates is None:
            return self.fullHistory
        funcExtractDate = lambda x: x.loc[dates] if x is not None else None
        return list(map(funcExtractDate, self.fullHistory))

    # Tranform synthetic surfaces as model data
    # Name of surfaces should be the date
    def convertRealDataToModelFormat(self, unformattedSurface):
        if (self.activateScaling):
            if (type(unformattedSurface) == type(list())) and (len(unformattedSurface) == 4):
                lambdaTransform = lambda x: x[0] if x[1] is None else x[1].transform(x[0])
                return list(map(lambdaTransform, zip(unformattedSurface, self.fullScaler)))
            elif (type(unformattedSurface) != type(list())):
                return self.volScaler.transform(unformattedSurface)
            else:
                raise ("Can not format as model data")
            return
        return unformattedSurface

    # Format data returned by a model to format
    # For instance variation are transformed as level with yesterday volatilities
    def formatModelDataAsDataSet(self, modelData):
        if (self.activateScaling):
            if (type(modelData) == type(list())) and (len(modelData) == 4):
                lambdaTransform = lambda x: x[0] if x[1] is None else x[1].inverse_transform(x[0])
                return list(map(lambdaTransform, zip(modelData, self.fullScaler)))
            elif (type(modelData) != type(list())):
                return self.volScaler.inverse_transform(modelData)
            else:
                raise ("Can not format as model data")
            return
        return modelData

    def scaleDataSets(self):
        if (self.activateScaling):
            # Define MinMax scaling for volatility
            self.volScaler = customMeanStdScale()  # customMinMaxScale()
            self.volScaler.fit(self.trainVol, enforceDataSetMin=0)  # Positive volatilities of course
            self.scaledTrainVol = self.volScaler.transform(self.trainVol)
            self.scaledTestVol = self.volScaler.transform(self.testVol)

            # Define MinMax scaling for coordinates
            self.coordinatesScaler = customMeanStdScale()  # customMinMaxScale()
            self.coordinatesScaler.fit(self.trainCoordinates, enforceDataSetMin=0)  # Positive volatilities of course
            self.scaledTrainCoordinates = self.coordinatesScaler.transform(self.trainCoordinates)
            self.scaledTestCoordinates = self.coordinatesScaler.transform(self.testCoordinates)

            # Define MinMax scaling for forward swap rates
            self.fwdScaler = customMeanStdScale()  # customMinMaxScale()
            self.fwdScaler.fit(self.trainFwd)
            self.scaledTrainFwd = self.fwdScaler.transform(self.trainFwd)
            self.scaledTestFwd = self.fwdScaler.transform(self.testFwd)
        else:
            self.scaledTrainVol = self.trainVol
            self.scaledTestVol = self.testVol

            self.scaledTrainCoordinates = self.trainCoordinates
            self.scaledTestCoordinates = self.testCoordinates

            self.scaledTrainFwd = self.trainFwd
            self.scaledTestFwd = self.testFwd
        return


def getATMDataFromPickle(dataSetPath,
                         trainingSetPercentage=0.8,
                         minStrikeIndex=0,
                         maturityStrikeIndex=0):
    with open(dataSetPath, "rb") as f:
        objectRead = pickle.load(f)

    def rankCalDays(dfDay):
        return dfDay["nBizDays"].rank()

    listRank = list(map(rankCalDays, objectRead))
    dfRank = pd.concat(listRank)
    dfConcat = pd.concat(objectRead)
    dfConcat["Rank"] = dfRank
    volDf = dfConcat.reset_index().set_index(["index", "Rank"]).drop(
        ["Date", "Forwards", "nBizDays", "nCalDays", "diff Days"], axis=1, errors="ignore").unstack()
    volDf.columns = volDf.columns.set_names("Moneyness", level=0)
    volDf = volDf.dropna(how="all", axis=1).astype("float64")

    fwdDf = dfConcat.reset_index().set_index(["index", "Rank"])["Forwards"].unstack()
    coordinatesRankDf = dfConcat.reset_index().set_index(["index", "Rank"])["nBizDays"].unstack()

    def bindBizDays(rows):
        bizDays = coordinatesRankDf.loc[rows.name].astype("float64")
        return pd.Series(list(zip(bizDays[rows.index.get_level_values("Rank")].values / 252.0,
                                  np.log(rows.index.get_level_values("Moneyness").astype("float64")))),
                         index=rows.index)

    coordinatesDf = volDf.apply(bindBizDays, axis=1)

    def getFwd(rowVol):
        ttmRank = rowVol.index.get_level_values("Rank")
        return pd.Series(fwdDf.loc[rowVol.name, ttmRank].values, index=rowVol.index)

    # Search for point in the vol dataframe the corresponding forward
    fwdDf = volDf.apply(getFwd, axis=1).dropna(how="all", axis=1).astype("float64")

    firstTestingDate = int(volDf.index.shape[0] * trainingSetPercentage)
    trainingDates = volDf.index[:firstTestingDate]

    trainVol = volDf.loc[trainingDates]
    testVol = volDf.drop(trainVol.index)
    trainVol = pd.DataFrame(trainVol.values, index=trainVol.index)
    testVol = pd.DataFrame(testVol.values, index=testVol.index)

    trainFwd = fwdDf.loc[trainVol.index]
    trainFwd = pd.DataFrame(trainFwd.values, index=trainFwd.index)[trainVol.columns]
    testFwd = fwdDf.drop(trainVol.index)
    testFwd = pd.DataFrame(testFwd.values, index=testFwd.index)[testVol.columns]

    testStrike = None
    trainStrike = None

    trainCoordinates = coordinatesDf.loc[trainingDates]
    trainCoordinates = pd.DataFrame(trainCoordinates.values, index=trainCoordinates.index)[trainVol.columns]
    testCoordinates = coordinatesDf.drop(trainVol.index)
    testCoordinates = pd.DataFrame(testCoordinates.values, index=testCoordinates.index)[testVol.columns]

    strikeDf = trainCoordinates.applymap(lambda x: x[1]).iloc[0]
    strikeList = np.sort(strikeDf.unique())
    minStrike = strikeList[minStrikeIndex]
    strikesKept = strikeDf[strikeDf >= minStrike].index

    maturityDf = trainCoordinates.applymap(lambda x: x[0]).iloc[0][strikesKept]
    maturityList = np.sort(maturityDf.unique())
    minMaturity = maturityList[minStrikeIndex]
    maturityKept = maturityDf[maturityDf >= minMaturity].index

    testVol = testVol[maturityKept]
    trainVol = trainVol[maturityKept]

    trainCoordinates = trainCoordinates[maturityKept]
    testCoordinates = testCoordinates[maturityKept]

    trainFwd = trainFwd[maturityKept]
    testFwd = testFwd[maturityKept]

    return testVol, trainVol, testFwd, trainFwd, testCoordinates, trainCoordinates, testStrike, trainStrike


def saveInterpolationResult(pathFile, paramDf, interpDf):
    pathTestFileInterp = pathFile + 'Interp'
    dictPickle = {}
    dictPickle["InterpParam"] = paramDf
    dictPickle["InterpolatedDf"] = interpDf
    with open(pathTestFileInterp, "wb") as f:
        pickle.dump(dictPickle, f, protocol=3)
    return


def FilterOnMaturity(Surface, coordinates, fwd, ttm):
    # Select (cube) data for maturity = ttm only
    expiries = coordinates.applymap(lambda x: x[0]) if type(coordinates) == type(
        pd.DataFrame()) else coordinates.map(lambda x: x[0])
    ListToKeep = [expiries.index[i] for i, x in enumerate(expiries) if x == ttm]
    maskedPoints = expiries.index.difference(ListToKeep)
    filteredCoordinates = pd.Series(True, index=expiries.index)
    filteredCoordinates.loc[maskedPoints] = False

    if Surface.ndim == 1:
        return Surface[filteredCoordinates], coordinates[filteredCoordinates], fwd[filteredCoordinates]
    else:
        return Surface[filteredCoordinates].dropna(how="any", axis=1), coordinates[filteredCoordinates].dropna(how="any", axis=1), fwd[filteredCoordinates].dropna(how="any", axis=1)


def FilterOnMoneynessRank(Surface, coordinates, fwd, rnk):
    # excludes monenyess in rnk list from ref_date data
    r = {}
    expiries = coordinates.applymap(lambda x: x[0]) if type(coordinates) == type(pd.DataFrame()) else coordinates.map(lambda x: x[0])

    # assumes unique ref_date, surface data
    expiriesList = stochasticModelQL.Remove(expiries)

    for ttm in expiriesList:
        _, df, _ = FilterOnMaturity(Surface, coordinates, fwd, ttm)
        r[ttm] = df.rank()
    listRank = pd.concat([r[x] for x in expiriesList], axis=0)
    listRank.sort_index()

    ListToKeep = [listRank.index[i] for i, x in enumerate(listRank) if x not in rnk]
    maskedPoints = coordinates.index.difference(ListToKeep)
    filteredCoordinates = pd.Series(True, index=coordinates.index)
    filteredCoordinates.loc[maskedPoints] = False

    return Surface[filteredCoordinates], coordinates[filteredCoordinates], fwd[filteredCoordinates], listRank


def FilterOnMaturityMoneneynessRank(Surface, coordinates, fwd, rnkMat, rnkMn):
    # excludes points where maturities in rnkMat AND  moneyness in rnkMn
    filteredSurface, filteredCoordinates, filteredFwd = removePointsWithInvalidCoordinatesWithFwd(Surface, coordinates, fwd)
    r1 = {}
    r2 = {}
    expiries = filteredCoordinates.applymap(lambda x: x[0]) if type(filteredCoordinates) == type(pd.DataFrame()) else coordinates.map(
        lambda x: x[0])

    for i, ref_date in enumerate(expiries.index):
        expiriesList = stochasticModelQL.Remove(expiries.loc[ref_date])
        r = {}
        for ttm in expiriesList:
            _, df, _ = FilterOnMaturity(filteredSurface.loc[ref_date], filteredCoordinates.loc[ref_date], filteredFwd.loc[ref_date], ttm)
            r[ttm] = df.rank()
        r1[ref_date] = pd.concat([r[x] for x in expiriesList], axis=0).sort_index()

        df2 = expiries.loc[ref_date].rank()
        r2[ref_date] = df2.rank(method='dense')

    listMoneynessRank = pd.DataFrame([r1[x] for x in expiries.index])
    listMaturityRank = pd.DataFrame([r2[x] for x in expiries.index])
    common_cols = np.intersect1d(listMoneynessRank.columns, listMaturityRank.columns)
    listMoneynessRank = listMoneynessRank[common_cols]
    listMaturityRank = listMaturityRank[common_cols]

    filteredMoneynessCoordinates = listMoneynessRank.applymap(lambda x: (x not in rnkMn)) if type(listMoneynessRank) == type(pd.DataFrame()) else listMoneynessRank.map(lambda x: (x not in rnkMn))
    filteredMaturityCoordinates = listMaturityRank.applymap(lambda x: (x not in rnkMat)) if type(listMaturityRank) == type(pd.DataFrame()) else listMaturityRank.map(lambda x: (x not in rnkMat))

    filteredRankCoordinates = filteredMoneynessCoordinates | filteredMaturityCoordinates

    if filteredSurface.ndim == 1:
        return filteredSurface[filteredRankCoordinates], filteredCoordinates[filteredRankCoordinates], filteredFwd[filteredRankCoordinates], listMoneynessRank, listMaturityRank, filteredRankCoordinates
    else:
        return Surface[filteredRankCoordinates].dropna(how="any", axis=1), filteredCoordinates[filteredRankCoordinates].dropna(how="any", axis=1), filteredFwd[filteredRankCoordinates].dropna(how="any", axis=1), listMoneynessRank, listMaturityRank, filteredRankCoordinates


def removePointsWithInvalidCoordinates(incompleteSurface, coordinates):
    # Filter location with incomplete observations
    def invalidCoordinates(x):
        if isinstance(x, tuple):
            return not any(np.isnan(x))
        return not np.isnan(x)

    if incompleteSurface.ndim == 1:
        filteredCoordinates = np.array(list(map(invalidCoordinates, coordinates)))
        return incompleteSurface[filteredCoordinates], coordinates[filteredCoordinates]
    else:
        filteredCoordinates = coordinates.applymap(invalidCoordinates) if type(coordinates) == type(pd.DataFrame()) else coordinates.map(invalidCoordinates)
        return incompleteSurface[filteredCoordinates].dropna(how="any", axis=1), coordinates[filteredCoordinates].dropna(how="any", axis=1)


def removePointsWithInvalidCoordinatesWithFwd(incompleteSurface, coordinates, fwd):
    # Filter location with incomplete observations
    def invalidCoordinates(x):
        if isinstance(x, tuple):
            return not any(np.isnan(x))
        return not np.isnan(x)

    if incompleteSurface.ndim == 1:
        filteredCoordinates = np.array(list(map(invalidCoordinates, coordinates)))
        return incompleteSurface[filteredCoordinates], coordinates[filteredCoordinates], fwd[filteredCoordinates]
    else:
        filteredCoordinates = coordinates.applymap(invalidCoordinates) if type(coordinates) == type(pd.DataFrame()) else coordinates.map(invalidCoordinates)
        return incompleteSurface[filteredCoordinates].dropna(how="any", axis=1), coordinates[filteredCoordinates].dropna(how="any", axis=1), fwd[filteredCoordinates].dropna(how="any", axis=1)


def readInterpolationResult(pathFile):
    pathTestFileInterp = pathFile + 'Interp'
    with open(pathTestFileInterp, "rb") as f:
        dictPickle = pickle.load(f)
    return dictPickle["InterpParam"], dictPickle["InterpolatedDf"]


class dataSetATMPickle(datasetATM):
    def __init__(self, pathToDataset,
                 trainingSetPercentage,
                 minExpiry,
                 completionRate,
                 scaleFeatures=False):

        self.nbMoneyness = 0
        self.MoneynessList = []
        self.nbTTM = 0
        self.ttmList = []
        self.minTTM = None
        self.isGridStable = False

        self.minStrike = 4
        self.minMaturity = 0
        self.logTransform = True

        super().__init__(pathToDataset,
                         trainingSetPercentage,
                         minExpiry,
                         completionRate,
                         scaleFeatures=scaleFeatures)

        listTokeep = [1.0, 2.0, 3.0, 4.0]
        self.setMaskedPoints(listTokeep)

    def datasetSummary(self):
        print("Number of days in dataset",
              self.getDataForModel()[0].shape[0])
        print("Number of days for testing", self.getTestingDataForModel()[0].shape[0])
        print("Number of days for training", self.getTrainingDataForModel()[0].shape[0])
        print("Working on Equity volatility level")
        print("Number of points in the grid : ", self.gridSize)
        print("Number of Moneyness : ", self.nbMoneyness)
        print("List : ", self.MoneynessList)
        print("Number of Time to maturities : ", self.nbTTM)
        print("List : ", self.ttmList)
        return

    def loadData(self):
        tmp = getATMDataFromPickle(self.pathToDataset, self.trainingSetPercentage, self.minStrike, self.minMaturity)

        self.testVol = tmp[0]
        self.trainVol = tmp[1]

        self.testCoordinates = tmp[4]
        self.trainCoordinates = tmp[5]

        self.testFwd = tmp[2]
        self.trainFwd = tmp[3]

        self.testStrike = tmp[6]
        self.trainStrike = tmp[7]

        def extractSingleton(df, coordIndex):
            valueList = np.unique(list(map(lambda x: x[coordIndex], np.ravel(df.values))))
            return valueList[~np.isnan(valueList)]

        fullCoordinatedDf = self.testCoordinates.append(self.trainCoordinates)
        self.MoneynessList = extractSingleton(fullCoordinatedDf, 1)
        self.ttmList = extractSingleton(fullCoordinatedDf, 0)
        self.nbMoneyness = self.MoneynessList.size
        self.nbTTM = self.ttmList.size
        self.gridSize = self.trainVol.columns.size

        return

    def setMaskedPoints(self, completionPoints):
        # self.maskedPoints = sampleSwaptionsToDelete(self.getTestingDataForModel(), 
        # completionRate)
        fullObs = self.getTestingDataForModel()[0].iloc[0]
        self.maskedPoints = fullObs.index.difference(completionPoints)

        # Matrix where True indicates that this point is completed (i.e. hidden on the market), false otherwise
        maskMatrix = pd.Series(False, index=fullObs.index)
        maskMatrix.loc[self.maskedPoints] = True
        self.maskSerie = maskMatrix
        # self.maskMatrix = maskMatrix.unstack(level=-1)

    # Return a deep copy with masked values
    def maskDataset(self, completeDataset):
        maskedRank = self.maskedPoints
        maskedDataset = completeDataset.copy()
        if completeDataset.ndim == 1:
            maskedDataset.loc[maskedRank] = np.NaN
        elif completeDataset.ndim == 2:
            maskedDataset[maskedRank] = np.NaN
        return maskedDataset

    # When the grid is not fixed - i.e. volatilities time to maturities are sliding -
    # we need to decide which instruments can be compared between two dates
    def decideInvestableInstruments(self):
        ttmDf = getTTMFromCoordinates(self.formatModelDataAsDataSet(self.getDataForModel()))

        pairIndexHistory = []  # series of pair of index
        nextTTMDf = ttmDf.shift(-1).dropna(how="all")
        for serie in ttmDf.head(-1).iterrows():
            currentDay = serie[1]
            nextDay = nextTTMDf.loc[serie[0]]
            currentRankForHedgeablePoints = currentDay[(currentDay - 1).isin(nextDay) & (~currentDay.isna())].index
            nextRankForHedgeablePoints = nextDay[(nextDay).isin(currentDay - 1) & (~nextDay.isna())].index
            if currentRankForHedgeablePoints.empty:  # case where current or day is not considered as a business day
                currentRankForHedgeablePoints = currentDay[(currentDay).isin(nextDay) & (~currentDay.isna())].index
                nextRankForHedgeablePoints = nextDay[(nextDay).isin(currentDay) & (~nextDay.isna())].index

            pairIndexHistory.append((currentRankForHedgeablePoints, nextRankForHedgeablePoints))
        # Last day
        pairIndexHistory.append((nextRankForHedgeablePoints, nextRankForHedgeablePoints))

        pairIndexHistory = pd.Series(pairIndexHistory, index=ttmDf.index)
        return pairIndexHistory
