# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 19:25:59 2017

@author: Sidney
Reference for emission and transition parameters
Emission = emissionParameters[word]["parameter"][tag]
transition = transitionParameters[prev_tag]["parameter"][current_tag]

"""
from collections import deque
import sys
import math 
from math import inf

sentimentSets = ["START","STOP","O","B-positive","I-positive","B-neutral","I-neutral","B-negative","I-negative"]

import pickle 
def save_obj(obj, fileDir, fileName ):
    with open('{0}\\variables\{1}.pkl'.format(fileDir,fileName),'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        
def load_obj(fileDir, fileName):
    with open('{0}\\variables\{1}.pkl'.format(fileDir,fileName), 'rb') as f:
        return pickle.load(f)

fileDir = "EN"
sentences = load_obj(fileDir,"sentences")
emissionParameters = load_obj(fileDir,"emissionParameters")
transitionParameters = load_obj(fileDir,"transitionParameters")

#TODO: Compute the sentences of the test-Set
def computeSentences(fileDir):
    with open('{0}\modifiedTest.txt'.format(fileDir), 'r',encoding='utf-8') as modTestSet:
        testSetString = modTestSet.read()
    sentences= []
    sentence = []
    testSetLines = testSetString.splitlines()
    for i in testSetLines:
        if (i != ''):
            #Valid ; choose "||" as delimiter
            sentence.append(i)
        else:
            #End of sentence reached
            sentences.append(sentence)
            sentence = []
    save_obj(sentences,fileDir,"sentences")

def computeTransitions(fileDir,tagCount):
    with open('{0}\modifiedTrain.txt'.format(fileDir), 'r',encoding='utf-8') as modTrainSet:
        trainSetString = modTrainSet.read()
    transitionParameters = {}
    y_prev = "START" 
    y_next = ""
    #Compute the emission counts
    #TODO: deal with the transition parameters
    #TODO: deal with anomalous data
    trainSetLines = trainSetString.splitlines(True)
    for i in trainSetLines:
        
        data = i.rsplit(" ",1)
        if(len(data)==2):
            word = data[0]
            tag = data[1].rstrip('\n')
            if(word == '' or tag not in sentimentSets):
                print("Corrupted data detected: {0}".format(i))
            else:
                y_next = tag
                nestedDictProcess(transitionParameters,y_prev,y_next)
                y_prev = tag              
                
        elif(i == '\n'):
            #print("Just a new line")
            y_next = "STOP"
            nestedDictProcess(transitionParameters,y_prev,y_next)
            y_prev = "START"
        else:
            print("Corrupted data detected: {0}".format(i))
        
    transParams = buildTransitionParameters(transitionParameters,tagCount)#Builds up the parameters
    return transParams

def nestedDictProcess(dictionary,key,subKey):
    if key not in dictionary:
        dictionary[key]={}
        dictionary[key]["count"] = {subKey:1}
    else:
        dictionary[key]["count"][subKey] = dictionary[key]["count"].get(subKey,0)+1 #Increment the count

def buildTransitionParameters(dictionary, tagCount):
    for y_prev, value in dictionary.items():   
        parameters = {}
        for y_next,count in value["count"].items():
            parameters[y_next] = count/tagCount[y_prev]#You must reference the count of the PREVIOUS TAG
        dictionary[y_prev]["parameters"] = parameters
    return dictionary


def aUV(transitionParameters,prev_tag,tag):
    dic = transitionParameters[prev_tag]["parameters"]
    return dic.get(tag,0)
     
def bVxi(emissionParameters,observation,tag):
    dic= emissionParameters[observation]["parameters"]
    return dic.get(tag,0)
    
def decodeAllSentences(sentences, fileDir, tP, eP):
    fileString = ""
    for sentence in sentences:
        fileString = fileString+viterbiAlgorithm(sentence,tP,eP)+"\n"
    with open('{0}\\dev.p3.out'.format(fileDir), 'w',encoding='utf-8') as outputFile:
        outputFile.write(fileString)
        
        
#TODO: Account for log-space
def viterbiAlgorithm(sentence_array, transitionParameters, emissionParameters):
    tP = transitionParameters
    eP = emissionParameters 
    terminalValue = 0
    tagSets = {"O":0,"B-positive":0,"I-positive":0,"B-neutral":0,"I-neutral":0,"B-negative":0,"I-negative":0}
    markovTable = []
    prev_tag = "START"
    
    trans =0
    emit =0
    #print("Commencing forward computation") 
    for i in range(0,len(sentence_array)):
        tagSets = {"O":0,"B-positive":0,"I-positive":0,"B-neutral":0,"I-neutral":0,"B-negative":0,"I-negative":0}
        markovTable.append(tagSets)
        observation = sentence_array[i]
        if i == 0:
            print("Base case")
            for tag in markovTable[i]:
                trans = aUV(tP,prev_tag,tag)
                emit = bVxi(eP,observation,tag)
                if(trans == 0 or emit ==0):
                    markovTable[i][tag] = -inf
                else:
                    markovTable[i][tag] = math.log10(emit*trans)
            #print(markovTable)
            
        else:
            #print("entering recursive case")
            for tag in markovTable[i]:
                values = []     
                #print(markovTable[i-1])
                for prev_tag in markovTable[i-1]:
                    prev_node_val = markovTable[i-1][prev_tag]
                    trans = aUV(tP,prev_tag,tag)
                    emit = bVxi(eP,observation,tag)
                    
                    #Check the values 
                    if(prev_node_val is -inf or trans == 0 or emit ==0):
                        tempVal = -inf
                    else:
                        #Need to reassign a new value based on the log space laws
                        #The value in the nodes are already in log base 10 
                        tempVal = prev_node_val+math.log10(trans*emit)
                    values.append(tempVal)
                #Set to None if no values available 
            
                markovTable[i][tag] = max(values)
                    

                
    #print("terminal case")
    lastTag = "STOP"
    values = [] 
    observation = sentence_array[-1]           
    for prev_tag in markovTable[-1]:
        prev_node_val = markovTable[-1][prev_tag]
        trans = aUV(tP,prev_tag,lastTag)
        #emit = bVxi(eP,observation,lastTag) STOP emits NOTHING
        #print("tag: {} trans: {}, prev: {}".format(prev_tag,trans, prev_node_val ))
        
        if(prev_node_val is -inf or trans == 0):
            tempVal = -inf
        else:
            #Need to reassign a new value based on the log space laws
            #The value in the nodes are already in log base 10 
            tempVal = prev_node_val+math.log10(trans)
        values.append(tempVal)
    terminalValue = max(values)
    
    #Backtracking
    print("Commencing back trekking with terminal value: {}".format(terminalValue))
    from collections import deque
    sequenceList = deque()
    latestTag = "STOP"
    for i in range(len(markovTable)-1, -1, -1):   
        observation = sentence_array[i]
        if(observation == "London"):
            print("Before: {}".format(markovTable[i]))

        for prev_tag,pi in markovTable[i].items():
            transition = aUV(tP,prev_tag,latestTag)
            if (transition ==0 or pi is -inf):
                markovTable[i][prev_tag] = -inf 
            else:
                markovTable[i][prev_tag] = pi+math.log10(transition)
        #validEntries = {prev_tag:pi for (prev_tag,pi) in markovTable[i].items() if pi is not None} 
        if(observation == "London"):
            print("After: {}".format(markovTable[i]))
        parent = max(markovTable[i], key=markovTable[i].get)
        #print("Parent found: {}".format(parent))
        sequenceList.appendleft(parent)
        latestTag = parent
        
    #Return the observation-state string
    obs_statePair = ""
    for i in range(0, len(sentence_array)):
        word = sentence_array[i]
        tag = sequenceList[i]
        obs_statePair = obs_statePair+"{0} {1}\n".format(word, tag)    
    return obs_statePair

            
            
                


trainingSets = ["EN","CN","FR","SG"]

"""
for i in trainingSets:
    fileDir = i
    tagCount = load_obj(fileDir,"tagCount")
    #transitionParameters = computeTransitions(fileDir,tagCount)
    #save_obj(transitionParameters,fileDir,"transitionParameters")
    computeSentences(fileDir)
"""

fileDir = "FR"
transitionParameters = load_obj(fileDir, "transitionParameters")   
emissionParameters = load_obj(fileDir,"emissionParameters")     
sentences = load_obj(fileDir,"sentences")
decodeAllSentences(sentences,fileDir,transitionParameters,emissionParameters)

    