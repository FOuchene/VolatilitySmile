import pandas as pd
import numpy as np
import dask
import scipy
import time

from functools import partial
from abc import ABCMeta, abstractmethod

from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from Code import point_in_polygon, factorialModel, loadData
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, DotProduct, WhiteKernel

import matplotlib.pyplot as plt
from scipy.interpolate import interp2d, griddata

#######################################################################################################
class InterpolationModel(factorialModel.FactorialModel):
    def __init__(self,
                 learningRate, 
                 hyperParameters, 
                 nbUnitsPerLayer, 
                 nbFactors,
                 modelName = "./bestInterpolationModel"):
        super().__init__(learningRate, 
                         hyperParameters,
                         nbUnitsPerLayer,
                         nbFactors,
                         modelName)
        

#ToolBox

#######################################################################################################
def getMaskedPoints(incompleteSurface, coordinates):
    return coordinates.loc[incompleteSurface.isna()]

def getMaskMatrix(incompleteSurface):
    maskMatrix = incompleteSurface.copy().fillna(True)
    maskMatrix.loc[~incompleteSurface.isna()] = False
    return maskMatrix

#maskedGrid : surface precising missing value with a NaN
#Assuming indexes and columns are sorted
#Select swaption coordinates (expiry, tenor) whose value is known and are on the boundary
#This defined a polygon whose vertices are known values
def selectPolygonOuterPoints(coordinates):
    outerPoints = []
    
    #Group coordinates by first coordinate
    splittedCoordinates = {}
    for tple in coordinates.values :
        if tple[0] not in splittedCoordinates :
            splittedCoordinates[tple[0]] = []
        splittedCoordinates[tple[0]].append(tple[1])
    
    #Get maximum and minimum for the second dimension
    for key in splittedCoordinates.keys():
        yMin = np.nanmin(splittedCoordinates[key])
        yMax = np.nanmax(splittedCoordinates[key])
        outerPoints.append((key,yMin))
        outerPoints.append((key,yMax))
        
    return outerPoints

def removeNaNcooridnates(coordinatesList):
    isNotNaN = [False if (np.isnan(x[0])  or np.isnan(x[1])) else True for x in coordinatesList]
    return coordinatesList[isNotNaN]

#Order a list of vertices to form a polygon 
def orderPolygonVertices(outerPointList):
    sortedPointList = np.sort(outerPointList) #np sort supports array of tuples 
    #Points are built as a pair of two points for value in the first dimension
    #Hence the polygon starts with points having the first value for the second dimension
    #(and order them along the first dimension)
    orderedListOfVertices = sortedPointList[::2]
    #We then browse the remaining points but in the reverse order for the second dimension
    orderedListOfVertices = sortedPointList[1::2][::-1]
    return orderedListOfVertices

#Select swaption coordinates (expiry, tenor) whose value is known and are on the boundary
#This defined a polygon whose vertices are known values
def buildInnerDomainCompletion(incompleteSurface, coordinates):
    coordinatesWithValues = coordinates.loc[~incompleteSurface.isna()]
    outerPointsList = selectPolygonOuterPoints(coordinatesWithValues)
    verticesList = orderPolygonVertices(outerPointsList)
    expiryVertices, tenorVectices = zip(*verticesList)
    return expiryVertices, tenorVectices

#Select swaption coordinates (expiry, tenor) whose value is known 
#and their coordinate corresponds to maximum/minimum value for x axis and y axis
#This defines a quadrilateral
def buildOuterDomainCompletion(incompleteSurface, coordinates):
    coordinatesWithValues = coordinates.loc[~incompleteSurface.isna()].values
    firstDimValues = list(map(lambda x : x[0], coordinatesWithValues))
    secondDimValues = list(map(lambda x : x[1], coordinatesWithValues))
    
    maxExpiry = np.amax(firstDimValues)
    minExpiry = np.nanmin(firstDimValues)
    
    maxTenor = np.amax(secondDimValues)
    minTenor = np.nanmin(secondDimValues)
    
    expiryVertices = [maxExpiry, maxExpiry, minExpiry, minExpiry, maxExpiry]
    tenorVectices = [maxTenor, minTenor, minTenor, maxTenor, maxTenor]
    
    return expiryVertices, tenorVectices

#verticesList : list of vertices defining the polygon
#Points : multiIndex serie for which we want to check the coordinates belongs to the domain defined by the polygon
#Use Winding number algorithm
def areInPolygon(verticesList, points):
    return pd.Series(points.map(lambda p : point_in_polygon.wn_PnPoly(p, verticesList) != 0).values,
                     index = points.index)

#Return the list (pandas Dataframe) of points which are located in the domain (as a closed set) 
#The closure ( i.e. edge of the domain ) is also returned
#defined by points which are not masked
def areInInnerPolygon(incompleteSurface, coordinates, showDomain = False):
    #Add the frontier
    gridPoints = coordinates.loc[~incompleteSurface.isna()]
    
    #Build polygon from the frontier
    expiriesPolygon, tenorsPolygon = buildInnerDomainCompletion(incompleteSurface, coordinates)
    polygon = list(zip(expiriesPolygon,tenorsPolygon))
    
    #Search among masked points which ones lie inside the polygon
    maskedPoints = getMaskedPoints(incompleteSurface, coordinates)
    interiorPoints = areInPolygon(polygon, maskedPoints)
    if not interiorPoints.empty :
        gridPoints = gridPoints.append(maskedPoints[interiorPoints]).drop_duplicates()
    
    
    if showDomain :
        plt.plot(expiriesPolygon,tenorsPolygon)
        plt.xlabel("First dimension")
        plt.xlabel("Second dimension")
        plt.plot(gridPoints.map(lambda x : x[0]).values,
                 gridPoints.map(lambda x : x[1]).values,
                 'ro')
        plt.show()
    
    return gridPoints
    
#Return the list (pandas Dataframe) of points which are located in the outer domain (as a closed set) 
#Outer domain is delimited by the maximum and minimum coordinates of the known values
#inner domain is delimited by the polygon whose vertices are the known points
#showDomain plots the boundary ( i.e. edge of the domain ) and the points which are inside the quadrilateral
def areInOuterPolygon(incompleteSurface, coordinates, showDomain = False):
    #Add the frontier
    gridPoints = coordinates.loc[~incompleteSurface.isna()]
    
    #Build polygon from the frontier
    expiriesPolygon, tenorsPolygon = buildOuterDomainCompletion(incompleteSurface, coordinates)
    polygon = list(zip(expiriesPolygon,tenorsPolygon))
    
    #Search among masked points which ones lie inside the polygon
    maskedPoints = getMaskedPoints(incompleteSurface, coordinates)
    interiorPoints = areInPolygon(polygon, maskedPoints)
    if not interiorPoints.empty :
        gridPoints = gridPoints.append(maskedPoints[interiorPoints]).drop_duplicates()
    
    
    if showDomain :
        plt.plot(expiriesPolygon,tenorsPolygon)
        plt.xlabel("First dimension")
        plt.xlabel("Second dimension")
        plt.plot(gridPoints.map(lambda x : x[0]).values,
                 gridPoints.map(lambda x : x[1]).values,
                 'ro')
        plt.show()
    
    return gridPoints

#######################################################################################################
#Linear interpolation with flat extrapolation
#Assume row are non empty
def interpolateRow(row, coordinates): 
    definedValues = row.dropna()
    if definedValues.size == 1 :
        return pd.Series(definedValues.iloc[0] * np.ones_like(row), 
                         index = row.index)
    else : 
        #Flat extrapolation and linear interpolation based on index (Tenor) value
        filledRow = row.interpolate(method='index', limit_direction = 'both')
        return filledRow
        
        

def formatCoordinatesAsArray(coordinateList):
    x = np.ravel(list(map(lambda x : x[0], coordinateList)))
    y = np.ravel(list(map(lambda x : x[1], coordinateList)))
    return np.vstack((x, y)).T

#Linear interpolation combined with Nearest neighbor extrapolation
# drawn from https://github.com/mChataign/DupireNN
def customInterpolator(interpolatedData, formerCoordinates, NewCoordinates):
    knownPositions = formatCoordinatesAsArray(formerCoordinates)
    
    xNew = np.ravel(list(map(lambda x : x[0], NewCoordinates)))
    yNew = np.ravel(list(map(lambda x : x[1], NewCoordinates)))
    # print(type(xNew))
    # print(type(yNew))
    # print(np.array((xNew, yNew)).T.shape)
    # print(type(interpolatedData))
    # print(type(knownPositions))
    # print()
    
    fInterpolation = griddata(knownPositions,
                              np.ravel(interpolatedData),
                              np.array((xNew, yNew)).T,
                              method = 'linear',
                              rescale=True)
    
    fExtrapolation =  griddata(knownPositions,
                               np.ravel(interpolatedData),
                               np.array((xNew, yNew)).T,
                               method = 'nearest',
                               rescale=True)
    
    return np.where(np.isnan(fInterpolation), fExtrapolation, fInterpolation)

def interpolate(incompleteSurface, coordinates):
    knownValues = incompleteSurface.dropna()
    knownLocation = coordinates.loc[knownValues.index]
    locationToInterpolate = coordinates.drop(knownValues.index)
    interpolatedValues = customInterpolator(knownValues.values, 
                                            knownLocation.values, 
                                            locationToInterpolate.values)
    completeSurface = pd.Series(interpolatedValues, 
                                index = locationToInterpolate.index).append(knownValues)
    
    return completeSurface.loc[incompleteSurface.index].rename(incompleteSurface.name)


def extrapolationFlat(incompleteSurface, coordinates):
    filteredSurface, filteredCoordinates = loadData.removePointsWithInvalidCoordinates(incompleteSurface, coordinates)
    correctedSurface = interpolate(filteredSurface, filteredCoordinates)
    correctedSurface = correctedSurface.append(pd.Series(incompleteSurface.drop(filteredCoordinates.index),
                                                         index = coordinates.drop(filteredCoordinates.index).index))
    return correctedSurface.sort_index()

#######################################################################################################
