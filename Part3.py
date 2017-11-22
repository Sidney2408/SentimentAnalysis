# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 19:25:59 2017

@author: Sidney
Reference for emission and transition parameters
Emission = emissionParameters[word]["parameter"][tag]
transition = transitionParameters[prev_tag]["parameter"][current_tag]
"""
from collections import deque

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
    sentence = ""
    testSetLines = testSetString.splitlines()
    for i in testSetLines:
        if (i != ''):
            #Valid ; choose "||" as delimiter
            sentence = sentence +i+"||"
        else:
            #End of sentence reached
            sentences.append(sentence.rstrip("||"))
            sentence = ""
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

def processSentences(sentences,transitionParameters, emissionParameters):
    for sentence in sentences:
        sentence_array = sentence.split("||")
        print(sentence_array)

def aUV(transitionParameters,prev_tag,tag):
    dic = transitionParameters[prev_tag]["parameters"]
    return dic.get(tag,0)
     

def bVxi(emissionParameters,observation,tag):
    dic= emissionParameters[observation]["parameters"]
    return dic.get(tag,0)
    
def decodeAllSentences(sentences, fileDir, tP, eP):
    fileString = ""
    for sentence in sentences:
        sentence_array = sentence.split("||")
        fileString = fileString+viterbiAlgorithm(sentence_array,tP,eP)+"\n"
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
    
    print("Commencing forward computation") 
    for i in range(0,len(sentence_array)):
        tagSets = {"O":0,"B-positive":0,"I-positive":0,"B-neutral":0,"I-neutral":0,"B-negative":0,"I-negative":0}
        markovTable.append(tagSets)
        if i == 0:
            observation = sentence_array[i]
            print("Base case")
            for tag in markovTable[i]:
                markovTable[i][tag] = aUV(tP,prev_tag,tag)*bVxi(eP,observation,tag)  
            print(markovTable)
            
        else:
            observation = sentence_array[i]
            print("entering recursive case")
            for tag in markovTable[i]:
                values = []     
                print(markovTable[i-1])
                for prev_tag in markovTable[i-1]:
                    #print(markovTable[i-1])
                    tempVal = markovTable[i-1][prev_tag]*aUV(tP,prev_tag,tag)*bVxi(eP,observation,tag)
                    values.append(tempVal)
                markovTable[i][tag] = max(values)

                
    print("terminal case")
    #You still have to iterate through every single dictionary 
    lastTag = "STOP"
    values = []            
    for tag in markovTable[len(markovTable)-1]:
        tempVal = markovTable[len(markovTable)-1][tag]*aUV(tP,tag,lastTag)
        values.append(tempVal)
    terminalValue = max(values)
    
    print("Commencing back trekking with terminal value: {}".format(terminalValue))
    from collections import deque
    
    sequenceList = deque()
    keyValue = terminalValue
    latestTag = "STOP"
    for i in range(len(markovTable)-1, -1, -1):   
        keyValPairs = {}
        for prev_tag,pi in markovTable[i].items():
            keyValPairs[prev_tag] = markovTable[i][prev_tag]*aUV(tP,prev_tag,latestTag)
        parent = max(keyValPairs, key=keyValPairs.get)
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
    transitionParameters = computeTransitions(fileDir,tagCount)
    save_obj(transitionParameters,fileDir,"transitionParameters")
"""
fileDir = "EN"
tagCount = load_obj(fileDir,"tagCount")          
transitionParameters = load_obj(fileDir, "transitionParameters")   
emissionParameters = load_obj(fileDir,"emissionParameters")     
sentences = load_obj(fileDir,"sentences")
decodeAllSentences(sentences,fileDir,transitionParameters,emissionParameters)
    