# -*- coding: utf-8 -*-

sentimentSets = ["Z","Y","X"]
filDir = "HW4"

import pickle 
def save_obj(obj, fileDir, fileName ):
    with open('{0}\\variables\{1}.pkl'.format(fileDir,fileName),'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        
def load_obj(fileDir, fileName):
    with open('{0}\\variables\{1}.pkl'.format(fileDir,fileName), 'rb') as f:
        return pickle.load(f)
    
def preprocess(fileDir):
    #Read the designated files first
    outliers = {} #TODO: Hunt for outliers and deal with them
    tagCount ={}
    trainWords={}
    modtrainWords = {}
    
    with open('{0}\\train.txt'.format(fileDir), 'r',encoding='utf-8') as trainSet:
        trainSetString = trainSet.read()
    
    print("Processing tagcounts and train word counts")
    #Parse through the training set
    trainSetLines = trainSetString.splitlines(True)
    #dictProcess(tagCount,"START")
    for i in trainSetLines:
        data = i.rsplit(" ",1)
        print(data)
        if(len(data)==2):
            word = data[0]
            tag = data[1].rstrip('\n')
            if(word == '' or tag not in sentimentSets):
                print("Corrupted data detected: {0}".format(i))
            else:
                dictProcess(tagCount, tag)#A helper function to tally up the counts 
                dictProcess(trainWords,word)
        elif(i == '\n'):
            dictProcess(tagCount,"START")
            #print("Just a new line")
            dictProcess(tagCount,"STOP")
        else:
            print("Corrupted data detected: {0}".format(i))                         
    return tagCount

def dictProcess(dictionary, key):
    dictionary[key] = dictionary.get(key,0)+1 


#Returns a dictionary with the emission parameters  
def computeEmissions(fileDir, tagCount):
    with open('{0}\\train.txt'.format(fileDir), 'r',encoding='utf-8') as modTrainSet:
        trainSetString = modTrainSet.read()
    emissionParameters = {}    
    #Compute the emission counts 
    trainSetLines = trainSetString.splitlines(True)
    for i in trainSetLines:
        data = i.rsplit(" ",1)
        if(len(data)==2):
            word = data[0]
            tag = data[1].rstrip('\n')
            if(word == '' or tag not in sentimentSets):
                print("Corrupted data detected: {0}".format(i))
            else:
                nestedDictProcess(emissionParameters,word,tag)#Builds up the dictionaries
        elif(i == '\n'):
            #print("Just a new line")
            pass
        else:
            print("Corrupted data detected: {0}".format(i))
            
    #Compute the observation parameters
    emitParams = buildEmissionParameters(emissionParameters,tagCount)#Builds up the parameters
    return emitParams

def computeTransitions(fileDir,tagCount):
    with open('{0}\\train.txt'.format(fileDir), 'r',encoding='utf-8') as modTrainSet:
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
    
#TODO: decide if you want to do {count,paramter} structure   
def nestedDictProcess(dictionary,key,subKey):
    if key not in dictionary:
        dictionary[key]={}
        dictionary[key]["count"] = {subKey:1}
    else:
        dictionary[key]["count"][subKey] = dictionary[key]["count"].get(subKey,0)+1 #Increment the count

def buildEmissionParameters(dictionary, tagCount):
    for key, value in dictionary.items():   
        parameters = {}
        for subKey,subvalue in value["count"].items():
            parameters[subKey] = subvalue/tagCount[subKey]
        dictionary[key]["parameters"] = parameters
    return dictionary

def buildTransitionParameters(dictionary, tagCount):
    for y_prev, value in dictionary.items():   
        parameters = {}
        for y_next,count in value["count"].items():
            parameters[y_next] = count/tagCount[y_prev]#You must reference the count of the PREVIOUS TAG
        dictionary[y_prev]["parameters"] = parameters
    return dictionary




#Here's where the real fun begins
#TODO: Compute the sentences of the test-Set
def computeSentences(fileDir):
    with open('{0}\\test.txt'.format(fileDir), 'r',encoding='utf-8') as modTestSet:
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
        
        

def viterbiAlgorithm(sentence_array, transitionParameters, emissionParameters):
    tP = transitionParameters
    eP = emissionParameters 
    terminalValue = 0
    #tagSets = {"O":0,"B-positive":0,"I-positive":0,"B-neutral":0,"I-neutral":0,"B-negative":0,"I-negative":0}
    tagSets = {"X":0,"Y":0,"Z":0}
    markovTable = []
    prev_tag = "START"
    
    print("Commencing forward computation")
    
    for i in range(0,len(sentence_array)):
        tagSets = {"X":0,"Y":0,"Z":0}
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
        print(i)
        keyValPairs = {}
        for prev_tag,pi in markovTable[i].items():
            keyValPairs[prev_tag] = markovTable[i][prev_tag]*aUV(tP,prev_tag,latestTag)
        parent = max(keyValPairs, key=keyValPairs.get)
        print("Parent found: {}".format(parent))
        sequenceList.appendleft(parent)
        latestTag = parent
        
    #Return the observation-state string
    obs_statePair = ""
    for i in range(0, len(sentence_array)):
        word = sentence_array[i]
        tag = sequenceList[i]
        obs_statePair = obs_statePair+"{0} {1}\n".format(word, tag)    
    return obs_statePair

fileDir = "HW4"
emitParams = load_obj(fileDir,"emitParams")
transParams = load_obj(fileDir,"transParams")
sentences = load_obj("HW4","sentences")
testSentence = sentences[0].split("||")
decodeAllSentences(sentences,fileDir,transParams,emitParams)

