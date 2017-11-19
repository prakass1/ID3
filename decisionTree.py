#!/usr/bin/python

'''

    File name: DecisionTree.py

    Team Members: 
        Nikhil Murthy (220641)
        Rutuja Pawar (220051)
        Subash Prakash (220408)

    Usage:

    To run the program, copy the input csv data file (car.data) in the same directory as the 
    program file, and issue the command "python DecisionTree.py". The output will be written to 
    the file, output.xml and will be saved in the same directory.
    
    Environment Tested:
    PYTHON : 3.6

    Description:    

   Generates an xml in a same directory as output.xml
    

    NOTE: Please have the input csv file (car.data) in the same folder as this file for execution.

'''

import math
import xml.etree.ElementTree as ET
import xml.dom.minidom as minidom
import sys


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
            valFreq[entry[i]] += 1
        else:
            valFreq[entry[i]]  = 1

    # Calculate the entropy of the data for the target attribute
    for freq in valFreq.values():
        dataEntropy += (-freq/len(data)) * math.log(freq/len(data), 4) 
        # log is 4 since we have 4 classifications 
    return dataEntropy

def gain(attributes, data, attr, targetAttr):

    valFreq = {}
    subsetEntropy = 0.0
    
    #find index of the attribute
    i = attributes.index(attr)

    # Calculate the frequency of each of the values in the target attribute
    for entry in data:
        if (entry[i] in valFreq):
            valFreq[entry[i]] += 1
        else:
            valFreq[entry[i]]  = 1

    # Calculate the sum of the entropy for each subset of records weighted
    # by their probability of occuring in the training set.
    for val in valFreq.keys():
        valProb        = valFreq[val] / sum(valFreq.values())
        dataSubset     = [entry for entry in data if entry[i] == val]
        subsetEntropy += valProb * entropy(attributes, dataSubset, targetAttr)


    # Subtract the entropy of the chosen attribute from the entropy of the
    # whole data set with respect to the target attribute (and return it)
    informationGain = (entropy(attributes, data, targetAttr) - subsetEntropy)
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
            major = key
            
    return major


#Reads the given input file
def readInputFile():
    data = []
    try:
        file = open('car1.csv','r')
        for line in file:
            line = line.strip("\r\n")
            data.append(line.split(","))
        file.close()
    except IOError:
        return -1
        
    return data

#Writes the output.xml file in the same folder
def writeOutputFile(tree):
    try:
        with open('output.xml','w') as fh:
            fh.write(tree)
        fh.close()

    except IOError:
        return -1
        
    return 0

#Pretty print the xml
def prettify(elem):
    string_element = ET.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(string_element)
    return reparsed.toprettyxml()

#getExamples
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

#Get values
def getValues(data,attributes,bestAttr):
    index = attributes.index(bestAttr)
    val = []
    for line in data:
        if line[index] not in val:
            val.append(line[index]) 
    return val

#   createtree method:In this if its the executing for the first time then we create a root node the we 
#    recursively call the gain method to for the sub-root 

def createTree(data,attributes,target,recur):
    
    recur= recur+1

    copy = data[:]
  
    vals = [record[attributes.index(target)] for record in data]

        
    default = majority(copy,attributes,target)
    
    #an xml element for root is created
    tree=ET.Element('tree')
    #if its the first xml then create a parent node with the entropy value and the classes
    
    if (recur == 1):
        i = attributes.index(target)
        valFreq={}
        for entry in copy:
            if (entry[i] in valFreq):
                valFreq[entry[i]] += 1
            else:
                valFreq[entry[i]]  = 1
        classes = ",".join(['%s:%s' % (key, value) for (key, value) in valFreq.items()])
        tree.set('entropy',str(entropy(attributes,copy,target)))
        tree.set("classes",classes)
        
 #If the dataset is empty or does not contain attributes then return a default value
    if not copy or (len(attributes) - 1) <= 0:
        return default

    #Return vals if they are of the same classification
    elif(vals.count(vals[0]) == len(vals)):
        return vals[0]

    #Here we creating sub node to the parent node with attribute which has highest information gain
    else:
        #Choose the best attribute 
        bestAttr = chooseAttribute(copy,attributes,target)
        tree_subroot=ET.Element('sub')
        
        for val in getValues(copy, attributes, bestAttr):
            node=ET.Element('node')
            node.set(bestAttr,str(val))
            splitExample = getExamples(copy,attributes,bestAttr,val)
            newAttributes = attributes[:]
            newAttributes.remove(bestAttr)

            valFreq={}
            index=newAttributes.index(target)

            for entry in splitExample:
                if(entry[index] in valFreq):
                    valFreq[entry[index]]+=1
                else:
                    valFreq[entry[index]]=1
                    classes = ",".join(['%s:%s' % (key, value) for (key, value) in valFreq.items()])
                    node.set('classes',classes)
                    node.set('entropy',str(entropy(splitExample, newAttributes, target)))
                    node.set(bestAttr, str(val))
                    tree_subroot.append(node)
            subtree = createTree(splitExample, newAttributes, target,recur)
            
            if isinstance(subtree, str):
                #This will be at the leaf level where the classification acceptable or unacceptable is made
                node.text = subtree
            else:
                for child in subtree:
                    node.append(child)
            #At the end, add the root(<tree>) at the top.
            if recur == 1:
                tree.append(node)
    
            if recur== 1:
                return prettify(tree)
            
        return tree_subroot
        
 

#####################
#Start of main      #
#####################


def main():

#Open the file and read Data
    data = readInputFile()
    
    if(data == -1):
        print("PLease check if the car.data is present in the folder where the script is run")
        sys.exit(-1)
#Taking the attributes in a list
    target='class'
    attributes = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']

#Calling ID3
    tree = createTree(data,attributes,target,0)
    
    returnCode = writeOutputFile(tree)
    if(returnCode == -1):
        print("Could not write to file, please check")
    else:
        print("Generated a Decision Tree and written it to output.xml \n ")
    
if __name__ == '__main__':
    main()
