import monkdata as m
import dtree
import drawtree_qt5 as draw
import numpy as np
import matplotlib.pyplot as plt
import random

entropyMonk1 = dtree.entropy(m.monk1)
entropyMonk2 = dtree.entropy(m.monk2)
entropyMonk3 = dtree.entropy(m.monk3)

print(f'Entropy for monk1: {entropyMonk1}')
print(f'Entropy for monk2: {entropyMonk2}')
print(f'Entropy for monk3: {entropyMonk3}')

informationGainMonk1 = list(map(lambda x: dtree.averageGain(m.monk1, x), m.attributes))
informationGainMonk2 = list(map(lambda x: dtree.averageGain(m.monk2, x), m.attributes))
informationGainMonk3 = list(map(lambda x: dtree.averageGain(m.monk3, x), m.attributes))

print(f'Information gain for all 6 attuributes for monk1: {informationGainMonk1}')
print(f'Information gain for all 6 attuributes for monk2: {informationGainMonk2}')
print(f'Information gain for all 6 attuributes for monk3: {informationGainMonk3}')

tree1 = dtree.buildTree(m.monk1, m.attributes)
print("monk1: Performance of decision tree on training set: {0} and on test set: {1}".format(dtree.check(tree1, m.monk1), dtree.check(tree1, m.monk1test)))

tree2 = dtree.buildTree(m.monk2, m.attributes)
print("monk2: Performance of decision tree on training set: {0} and on test set: {1}".format(dtree.check(tree2, m.monk2), dtree.check(tree2, m.monk2test)))

tree3 = dtree.buildTree(m.monk3, m.attributes)
print("monk3: Performance of decision tree on training set: {0} and on test set: {1}".format(dtree.check(tree3, m.monk3), dtree.check(tree3, m.monk3test)))

def bestPrunedFromList(tree, validationDataset):
    listOfTrees = dtree.allPruned(tree)
    bestValue = dtree.check(tree, validationDataset)
    bestTree = listOfTrees[len(listOfTrees)-1]
    for tree in listOfTrees:
        temp = dtree.check(tree, validationDataset)
        if temp > bestValue:
            bestValue = temp
            bestTree = tree
    return bestTree

def bestPruned(tree, validationDataset):
    bestTree = tree
    while True:
        tempTree = bestPrunedFromList(bestTree, validationDataset);
        if (dtree.check(tempTree, validationDataset) >= dtree.check(bestTree, validationDataset)):
            bestTree = tempTree
        else:
            return bestTree

def partition(data, fraction):
    ldata = list(data)
    random.shuffle(ldata)
    breakPoint = int(len(ldata) * fraction)
    return ldata[:breakPoint], ldata[breakPoint:]

fractions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

def getErrorInDataset(dataset, testset, fraction):
    errorsInTrainingSet = []
    errorsInTestSet = []
    for i in range(100):
        trainingSet, validationSet = partition(dataset, fraction)
        trainedTree = dtree.buildTree(trainingSet, m.attributes)
        bestTree = bestPrunedFromList(trainedTree, validationSet)
        errorsInTrainingSet.append(1 - dtree.check(bestTree, dataset))
        errorsInTestSet.append(1 - dtree.check(bestTree, testset))
    return [np.mean(errorsInTrainingSet), np.std(errorsInTrainingSet), np.mean(errorsInTestSet), np.std(errorsInTestSet)]

errorsInMonk1 = list(map(lambda fraction: getErrorInDataset(m.monk1, m.monk1test, fraction), fractions));
errorsInMonk3 = list(map(lambda fraction: getErrorInDataset(m.monk3, m.monk3test, fraction), fractions));

meanErrorInTestingSetMonk1 = [error[2] for error in errorsInMonk1]
stdErrorInTestingSetMonk1 = [error[3] for error in errorsInMonk1]
meanErrorInTestingSetMonk3 = [error[2] for error in errorsInMonk3]
stdErrorInTestingSetMonk3 = [error[3] for error in errorsInMonk3]

print(f'Lowest test error for fraction {fractions[meanErrorInTestingSetMonk1.index(min(meanErrorInTestingSetMonk1))]} for monk1: {min(meanErrorInTestingSetMonk1)}')
print(f'Lowest test error for fraction {fractions[meanErrorInTestingSetMonk3.index(min(meanErrorInTestingSetMonk3))]} for monk3: {min(meanErrorInTestingSetMonk3)}')

fig, axes = plt.subplots(1, 2, num="Mean and Standard Deviation of errors for test dataset for monk 1 and monk3", figsize=(15, 5))
errorsInMonk1Fig, errorsInMonk3Fig = axes
errorsInMonk1Fig.errorbar(fractions, meanErrorInTestingSetMonk1, stdErrorInTestingSetMonk1, linestyle='None', marker='^', label="monk1 error mean and std", color="red")
errorsInMonk1Fig.legend()
errorsInMonk1Fig.set_xlabel('Fractions')
errorsInMonk1Fig.set_ylabel('Error')
errorsInMonk3Fig.errorbar(fractions, meanErrorInTestingSetMonk3, stdErrorInTestingSetMonk3, linestyle='None', marker='*', label="monk3 error mean and std", color="orange")
errorsInMonk3Fig.legend()
errorsInMonk3Fig.set_xlabel('Fractions')
errorsInMonk3Fig.set_ylabel('Error')
plt.show()
