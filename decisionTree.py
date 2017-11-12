#!/usr/bin/python

import math
from xml.etree.ElementTree import Element, SubElement, Comment, tostring


#Calculates the entropy of the given data set for the target attr
def entropy(attributes, data, targetAttr):

    valFreq = {}
    dataEntropy = 0.0
    
    #find index of the target attribute
    i = 0
    for entry in attributes:
        if (targetAttr == entry):
            break
        ++i
    
    # Calculate the frequency of each of the values in the target attribute
    for entry in data:
        if (entry[i] in valFreq):
            valFreq[entry[i]] += 1.0
        else:
            valFreq[entry[i]]  = 1.0

    # Calculate the entropy of the data for the target attribute
    for freq in valFreq.values():
        dataEntropy += (-freq/len(data)) * math.log(freq/len(data), 2) 
    print("ENtropy:: " + str(dataEntropy) + "\n")    
    return dataEntropy

def gain(attributes, data, attr, targetAttr):
    """
    Calculates the information gain (reduction in entropy) that would
    result by splitting the data on the chosen attribute (attr).
    """
    valFreq = {}
    subsetEntropy = 0.0
    
    #find index of the attribute
    i = attributes.index(attr)

    # Calculate the frequency of each of the values in the target attribute
    for entry in data:
        if (entry[i] in valFreq):
            valFreq[entry[i]] += 1.0
        else:
            valFreq[entry[i]]  = 1.0
    # Calculate the sum of the entropy for each subset of records weighted
    # by their probability of occuring in the training set.
    for val in valFreq.keys():
        valProb        = valFreq[val] / sum(valFreq.values())
        dataSubset     = [entry for entry in data if entry[i] == val]
        subsetEntropy += valProb * entropy(attributes, dataSubset, targetAttr)

    # Subtract the entropy of the chosen attribute from the entropy of the
    # whole data set with respect to the target attribute (and return it)
    informationGain = (entropy(attributes, data, targetAttr) - subsetEntropy)
    print("Information Gain is:: " + str(informationGain) + "\n")
    return informationGain

#choose best attribute 
def chooseAttribute(data, attributes, target):
    best = attributes[0]
    maxGain = 0
    for attr in attributes:
        newGain = gain(attributes, data, attr, target) 
        if newGain > maxGain:
            maxGain = newGain
            best = attr
    return best

#find most common value for an attribute
def majority(data, attributes, target):
    #find target attribute
    valFreq = {}
    #find target in data
    index = attributes.index(target)
    #calculate frequency of values in target attr
    for tuple in data:
        if (tuple[index] in valFreq):
            valFreq[tuple[index]] += 1 
        else:
            valFreq[tuple[index]] = 1
    max = 0
    major = ""
    for key in valFreq.keys():
        if valFreq[key]>max:
            max = valFreq[key]
            print("Value Frequency: " + str(max) + "\n")
            major = key
            
    return major


########################################
#Name: readInputFile(file)             #
#Description: read a input data and    #
#             puts it into a list      #
########################################
def readInputFile(file):
    data = []
    for line in file:
        line = line.strip("\r\n")
        data.append(line.split(","))
    file.close()
    return data


###############################
#Name:getExamples
#
#
###############################
def getExamples(data, attributes, best, val):
    examples = []
    index = attributes.index(best)
    for entry in data:
        #find entries with the give value
        if (entry[index] == val):
            newEntry = []
            #add value if it is not in best column
            for i in range(0,len(entry)):
                if(i != index):
                    newEntry.append(entry[i])
            examples.append(newEntry)
    return examples

###############################
#Name: getValues()
#
################################
def getValues(data,attributes,bestAttr):
    index = attributes.index(bestAttr)
    val = []
    for line in data:
        if line[index] not in val:
            val.append(line[index]) 
    return val

############################################
# Name:                                    #
# Description:                             #
#                                          #
#                                          #
############################################
def createTree(data,attributes,target,recur):
    recur= recur + 1
    #print(target)
    #print(attributes[6])
    #Copy of the list into a new list
    copy = data[:]
    vals = []
    for record in copy:
        vals = record[attributes.index(target)]
        
    default = majority(copy,attributes,target)
    
    
    
    
    #If the dataset is empty or does not contain attributes then return a default value
    if not copy or (len(attributes) - 1) <= 0:
        return default
    
    #Return vals if they are of the same classification
    elif(vals.count(vals[0]) == len(vals)):
        return vals[0]
    
    else:
        #Choose the best attribute
        bestAttr = chooseAttribute(copy,attributes,target)
        tree = {bestAttr:{}}
        
        for val in getValues(copy, attributes, bestAttr):
            splitExample = getExamples(copy,attributes,bestAttr,val)
            newAttributes = attributes[0:]
            newAttributes.remove(bestAttr)
            subTree = createTree(splitExample, newAttributes, target,recur)
            tree[bestAttr][val] = subTree
            
        
    
    return tree

#####################
#Start of main      #
#####################
debug=True

#open the file for insertion
file = open('car1.data','r')

#Data is being read
data = readInputFile(file)

#Printing data
if(debug):
    for line in data:
        print(line)

#Taking the attributes in a list
target='class'
attributes = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']

#Calling ID3
tree = createTree(data,attributes,target,0)

print("Tree")
print(str(tree))
