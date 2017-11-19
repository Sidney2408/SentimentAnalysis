# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 15:09:39 2017

@author: Sidney
"""
testDict = {'完美': {'B-positive': {'count': 1}, 'O': {'count': 1}}, '幸福': {'O': {"count": 0}}}
tagCount = {'B-positive':1,'O':2, 'B-negative':3}

parameter = "exy"
    
#max(testDict['a']["parameters"], key=testDict['a']["parameters"].get)
        
def buildParameters(dictionary, tagCount):
    for key, value in dictionary.items():   
        parameters = {}
        for subKey,subvalue in value.items():
            parameters[subKey] = subvalue["count"]/tagCount[subKey]
        dictionary[key]["parameters"] = parameters
    return dictionary

    

#Backup function  
"""
def nestedDictProcess(dictionary,key,subKey):
    if key not in dictionary:
        dictionary[key]={}
        dictionary[key][subKey] = {"count":0}#The 0 is wrong
    else:
        if subKey not in dictionary[key]:
            dictionary[key][subKey] = {"count":0}
        else:
            dictionary[key][subKey]["count"] = dictionary[key][subKey].get("count",0)+1 #Increment the count

def buildParameters(dictionary, tagCount):
    for key, value in dictionary.items():   
        parameters = {}
        for subKey,subvalue in value.items():
            parameters[subKey] = subvalue["count"]/tagCount[subKey]
        dictionary[key]["parameters"] = parameters
    return dictionary  
    
"""
#TODO: decide if you want to do {count,paramter} structure   
def nestedDictProcess(dictionary,key,subKey):
    if key not in dictionary:
        dictionary[key]={}
        dictionary[key]["count"] = {subKey:1}
    else:
        dictionary[key]["count"][subKey] = dictionary[key]["count"].get(subKey,0)+1 #Increment the count

def buildParameters(dictionary, tagCount):
    for key, value in dictionary.items():   
        parameters = {}
        for subKey,subvalue in value["count"].items():
            parameters[subKey] = subvalue/tagCount[subKey]
        dictionary[key]["parameters"] = parameters
    return dictionary


sentimentSets = ["START","STOP","O","B-positive","I-positive","B-neutral","I-neutral","B-negative","I-negative"]


fileName = "EN"

def detectAnomalies(fileDir):
    with open('{0}\modifiedTrain.txt'.format(fileDir), 'r',encoding='utf-8') as modTrainSet:
        trainSetString = modTrainSet.read()
    trainSetLines = trainSetString.splitlines(True)
    sentences = 0
    index = 0
    indices = ""
    for i in trainSetLines:
        index+= 1
        data = i.split(" ")
        if(len(data)==2):
            word = data[0]
            tag = data[1].rstrip('\n')
            if(word == '' or tag not in sentimentSets):
                print("Corrupted data detected: {0}".format(i))            
                indices= indices +"{0} {1}\n".format(i,index)
        elif(i == '\n'):
            sentences +=1
        else:
            print("Corrupted data detected: {0}".format(i))
            indices= indices +"{0} {1}\n".format(i,index)
    indices= "Sentences {0}".format(sentences)+indices
    with open('{0}\\variables\errors.txt'.format(fileDir), 'w',encoding='utf-8') as outputFile:
        outputFile.write(indices)

detectAnomalies("SG")








