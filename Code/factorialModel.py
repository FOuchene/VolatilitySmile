import os

tmpFolder = "./tmp/"


class FactorialModel:
    #######################################################################################################
    # Construction functions
    #######################################################################################################
    def __init__(self,
                 learningRate,
                 hyperParameters,
                 nbUnitsPerLayer,
                 nbFactors,
                 modelName="./bestFactorialModel"):
        # Hyperparameters
        self.learningRate = learningRate
        self.hyperParameters = hyperParameters
        self.nbUnitsPerLayer = nbUnitsPerLayer
        self.nbFactors = nbFactors
        self.layers = []
        self.nbEncoderLayer = 0
        self.batchSize = -1
        self.lossHolderExponent = (
            hyperParameters["lossHolderExponent"] if "lossHolderExponent" in hyperParameters else 4)

        pathToModels = os.path.join(tmpFolder, 'modelSaved')  # os.getcwd() + "\\..\\modelSaved\\"#"..\\modelSaved\\"
        if not os.path.isdir(pathToModels):
            os.mkdir(pathToModels)
        self.metaModelName = os.path.normpath(
            os.path.join(pathToModels, os.path.normpath(modelName) + ".cpkt")).replace("\\", "/")
        self.metaModelNameInit = os.path.normpath(
            os.path.join(pathToModels, os.path.normpath(modelName) + "Init" + ".cpkt")).replace("\\", "/")
        self.verbose = (('verbose' in hyperParameters) & hyperParameters['verbose'])

        if self.verbose:
            print("Create tensor for training")
        self.buildModel()

        self.variationMode = False  # activate learning with variations instead of

    # Build the architecture, losses and optimizer.
    def buildModel(self):
        raise NotImplementedError("Abstract Class !")
        return None

    #######################################################################################################
    # Training functions
    #######################################################################################################

    # Sample Mini-batch: data loader to be implemented
    def splitValidationAndTrainingSet(self, dataSetTrainList):
        # Sample days in validation set
        percentageForValidationSet = self.hyperParameters['validationPercentage'] if (
                    'validationPercentage' in self.hyperParameters) else 0.2
        nbValidationSetDays = int(percentageForValidationSet * dataSetTrainList[0].index.size)
        validationSetDays = dataSetTrainList[0].index[-nbValidationSetDays:]
        validationDataSetList = [x.loc[validationSetDays].sort_index() if x is not None else None for x in
                                 dataSetTrainList]
        trainingDataSetList = [x.drop(validationSetDays).sort_index() if x is not None else None for x in
                               dataSetTrainList]
        return validationDataSetList, trainingDataSetList
