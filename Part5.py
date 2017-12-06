# -*- coding: utf-8 -*-

from collections import deque
import math 
from math import inf
import pickle

#part 5 changes much of part2-4 code to prevent writing to files. Instead, strings are passed around in RAM (might be an issue)
#Because of this, you must run the entire code from the beginning to completion even if you stopped halfway previously
#Note: This file takes damn long to run (about 30 minutes).


#CONSTANTS, change here for settings

TEST=True
#if TEST is true, read and write from test.in. Otherwise, it will be from dev.in
#writes to dev.p5.out (or test.p5.out)
#should not write to anything else

fileDir = "EN"
#Specifies language to use. choose between EN and FR

NUMOFITER = 10
#Number of times to run perceptron. 3 is fine, 10 takes a while
TRAIN = True
#To train the model on train.txt. If false, loads from pickle file

### PART 4 modified
def part4Vit(sentences, fileDir, tP, eP):
    #Runs maxMarginal on all sentences, and writes to test.p5.out (or dev.p5.out)
    fileString = ""
    for sentence in sentences:
        maxMarginal(sentence,tP,eP)
        fileString = fileString+maxMarginal(sentence,tP,eP)+"\n"
    
    if TEST:    
        with open('{0}\\test.p5.out'.format(fileDir), 'w',encoding='utf-8') as outputFile:
            outputFile.write(fileString)
    else:
        with open('{0}\\dev.p5.out'.format(fileDir), 'w',encoding='utf-8') as outputFile:
            outputFile.write(fileString)
            
def maxMarginal(sentence,tP,eP):
    n = len(sentence)
    alpha = deque()
    firstTag = "START"

    for index in range(0,n):
        observation = sentence[index] 
        tagSets = {"O":0,"B-positive":0,"I-positive":0,"B-neutral":0,"I-neutral":0,"B-negative":0,"I-negative":0}
        alpha.append(tagSets)
        
        if (index == 0):
            #Base case
            for current_tag in alpha[index]:
                #print(current_tag)
                transition = aUV(tP,firstTag,current_tag)
                alpha[0][current_tag] = transition
        else:
            #Iterate over every tag in the current set
            previous_observation = sentence[index-1] 
            for current_tag in alpha[index]:
                runningTotal = 0                
                #Summation over the previous set 
                for previous_tag in alpha[index-1]:
                    alpha_u_n = alpha[index-1][previous_tag]
                    trans = aUV(tP,previous_tag,current_tag)
                    emit = bVxi(eP,previous_observation, previous_tag)
                    runningTotal = runningTotal + alpha_u_n*trans*emit
                    if(runningTotal < 1e-300 and runningTotal!= 0):
                        print("Danger! {}".format(runningTotal))

                alpha[index][current_tag] = runningTotal
                    

        
    beta = deque()
    lastTag = "STOP"
    
    for i in range(0,n):
        tagSets = {"O":0,"B-positive":0,"I-positive":0,"B-neutral":0,"I-neutral":0,"B-negative":0,"I-negative":0}#Forgot to refresh the array
        beta.append(tagSets)
       
    for index in range(n-1,-1,-1):
        observation = sentence[index]
         
        if (index == n-1):
            #base case 
            for current_tag in beta[index]:
                transition = aUV(tP,current_tag,lastTag)
                emission = bVxi(eP,observation,current_tag)
                #print("word: {} tag:{} trans: {}, emiss: {}".format(observation, current_tag, transition,emission))
                beta_u_n = transition*emission
                beta[index][current_tag] = beta_u_n
                #print(beta[0])                    
        else:
            for current_tag in beta[index]:
                runningTotal = 0
                for previous_tag in beta[index+1]:
                    beta_u_n = beta[index+1][previous_tag]
                    trans = aUV(tP,current_tag,previous_tag)#Reversed flow 
                    emit = bVxi(eP,observation, current_tag)
                    runningTotal = runningTotal+ beta_u_n*trans*emit
                    
                    if(runningTotal < 1e-300 and runningTotal!= 0):
                        print("Danger! {}".format(runningTotal))
                        
                beta[index][current_tag] = runningTotal
        
    obs_statePair = ""   
    for index in range(0,n):
        word = sentence[index]
        dict_alpha = alpha[index]
        dict_beta =  beta[index]
        #alpha_x_beta = {k:dict_alpha[k]*dict_beta[k] for k in tagSets}
        alpha_x_beta = {}
        for sentiment in tagSets:
            alpha_x_beta[sentiment] = dict_alpha[sentiment]*dict_beta[sentiment]
        tag = max(alpha_x_beta, key=alpha_x_beta.get)
        #print("word: {}, tag: {}".format(word,tag))
        obs_statePair = obs_statePair +word +" "+tag +"\n"
    return obs_statePair
     
def aUV(transitionParameters,prev_tag,tag):
    dic = transitionParameters[prev_tag]["parameters"]
    return dic.get(tag,0)
     
def bVxi(emissionParameters,observation,tag):
    dic= emissionParameters[observation]["parameters"]
    return dic.get(tag,0)

### END PART 4



### PART 2 mod

sentimentSets = ["START","STOP","O","B-positive","I-positive","B-neutral","I-neutral","B-negative","I-negative"]
def preprocess(fileDir,kVal):
    #returns tuple of (tagCount,cleanedTrainString,cleanedTestString)
    #Read the designated files first
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
        if(len(data)==2):
            word = data[0]
            tag = data[1].rstrip('\n')
            if(word == '' or tag not in sentimentSets):
                print("Corrupted data detected: {0}").format(i)
            else:
                dictProcess(tagCount, tag)#A helper function to tally up the counts 
                dictProcess(trainWords,word)
        elif(i == '\n'):
            dictProcess(tagCount,"START")
            #print("Just a new line")
            dictProcess(tagCount,"STOP")
        else:
            print("Corrupted data detected: {0}".format(i))
    
    
    print("Replacing words in the training set that appear less than k times with #UNK#") 
    #Replace the words in the training set that appear less than k times with #UNK#
    wordDict = {k:v for (k,v) in trainWords.items() if v < kVal} 
    modifiedString = ""
    for i in trainSetLines:
        data = i.rsplit(" ",1)
        #print(data)
        #What about cases where there is a word without a sentiment?
        #TODO: account for cases where there is corrupted data.
        if(len(data)==2):
            word = data[0]
            tag = data[1].rstrip('\n')
            if(word in wordDict):
                words = "#UNK# "+tag+"\n"
                modifiedString = modifiedString+ words  
            elif(word not in wordDict):
                #print("Word not in the dictionary just add as usual: {0}".format(i))
                modifiedString = modifiedString+i
            else:
                print("I have no idea what this is: {0}").format(i)
                modifiedString = modifiedString+i
        elif(i == '\n'):
            #print("Just a new line")
            modifiedString = modifiedString+i
        else:
            print("Corrupted data detected: {0}".format(i))
            modifiedString = modifiedString+i
    
    #Building modtrainWords:
    #TODO: find a way to streamline this computation
    for i in modifiedString.splitlines(True):
        data = i.rsplit(" ",1)
        if(len(data)==2):
            word = data[0]
            tag = data[1].rstrip('\n')
            if(word == '' or tag not in sentimentSets):
                print("Corrupted data detected: {0}").format(i)
            else:
                dictProcess(modtrainWords,word)
        elif(i == '\n'):
            #print("Just a new line")
            pass
        else:
            print("Corrupted data detected: {0}".format(i))        
    
    #Reading the words inside the testSet that do not appear in the training set
    testWords = {}
    testSetString=""
    if TEST:
        with open('{0}\\test.in'.format(fileDir), 'r',encoding='utf-8') as testSet:
            testSetString = testSet.read()
    else:
        with open('{0}\\dev.in'.format(fileDir), 'r',encoding='utf-8') as testSet:
            testSetString = testSet.read()
            
    testSetLines = testSetString.splitlines()#This converts all the '\n' to ''
    for i in testSetLines:
        if(i!=''):
            dictProcess(testWords,i)
    
    wordsNotInTrainingSet = set(testWords) - set(modtrainWords)
    
    modifiedTestString = ""
    
    for i in testSetLines:
        if (i != ''):
            if i in wordsNotInTrainingSet:
                modifiedTestString = modifiedTestString+"#UNK#\n"
            else:
                modifiedTestString = modifiedTestString+ i+'\n'
        else:
            modifiedTestString = modifiedTestString+ '\n'
    
    return (tagCount,modifiedString,modifiedTestString)
#    with open('{0}\modifiedTest.txt'.format(fileDir), 'w',encoding='utf-8') as outputTestFile:
#        outputTestFile.write(modifiedTestString)      
#    
#    
#    with open('{0}\modifiedTrain.txt'.format(fileDir), 'w',encoding='utf-8') as outputFile:
#        outputFile.write(modifiedString)
#    
  

def dictProcess(dictionary, key):
    dictionary[key] = dictionary.get(key,0)+1 


#Returns a dictionary with the emission parameters  
def computeEmissions(fileDir, tagCount, modifiedTrainingString):
#    with open('{0}\modifiedTrain.txt'.format(fileDir), 'r',encoding='utf-8') as modTrainSet:
#        trainSetString = modTrainSet.read()
    trainSetString = modifiedTrainingString
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


def save_obj(obj, fileDir, fileName ):
    with open('{0}\\variables\{1}.pkl'.format(fileDir,fileName),'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        

def load_obj(fileDir, fileName):
    with open('{0}\\variables\{1}.pkl'.format(fileDir,fileName), 'rb') as f:
        return pickle.load(f)
    


def detectAnomalies(fileDir):
    with open('{0}\modifiedTrain.txt'.format(fileDir), 'r',encoding='utf-8') as modTrainSet:
        trainSetString = modTrainSet.read()
    trainSetLines = trainSetString.splitlines(True)
    sentences = 0
    index = 0
    indices = ""
    for i in trainSetLines:
        index+= 1
        data = i.split(" ")#Such was a wrong way of splitting
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

### END PART 2
            
            
### PART 3

sentimentSets = ["START","STOP","O","B-positive","I-positive","B-neutral","I-neutral","B-negative","I-negative"]

#TODO: Compute the sentences of the test-Set
def computeSentences(fileDir,cleanedTestString):
#    with open('{0}\modifiedTest.txt'.format(fileDir), 'r',encoding='utf-8') as modTestSet:
#        testSetString = modTestSet.read()
    testSetString=cleanedTestString
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
    return sentences
    #save_obj(sentences,fileDir,"sentences")

def computeTransitions(fileDir,tagCount,modifiedTrainingString):
#    with open('{0}\modifiedTrain.txt'.format(fileDir), 'r',encoding='utf-8') as modTrainSet:
#        trainSetString = modTrainSet.read()
    trainSetString = modifiedTrainingString
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

def buildTransitionParameters(dictionary, tagCount):
    for y_prev, value in dictionary.items():   
        parameters = {}
        for y_next,count in value["count"].items():
            parameters[y_next] = count/tagCount[y_prev]#You must reference the count of the PREVIOUS TAG
        dictionary[y_prev]["parameters"] = parameters
    return dictionary


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
            #print("Base case")
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
    #print("Commencing back trekking with terminal value: {}".format(terminalValue))
    sequenceList = deque()
    latestTag = "STOP"
    for i in range(len(markovTable)-1, -1, -1):   
        observation = sentence_array[i]

        for prev_tag,pi in markovTable[i].items():
            transition = aUV(tP,prev_tag,latestTag)
            if (transition ==0 or pi is -inf):
                markovTable[i][prev_tag] = -inf 
            else:
                markovTable[i][prev_tag] = pi+math.log10(transition)
        #validEntries = {prev_tag:pi for (prev_tag,pi) in markovTable[i].items() if pi is not None} 
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
            
### END PART 3



### PART 5
#Part 2 cleans train and test data. Test data is further parsed in part 3, into 'sentences'
#Otherwise, test data ('sentences') is not touched, and train data is used to generate emis and trans parameters
#part 3 also uses viterbi to train the params. We modify this to train perceptrons



if TRAIN:
    tagCount,modifiedTrainingString,modifiedTestString = preprocess(fileDir,3)
    #save_obj(tagCount, fileDir, "tagCount")
    emissionParameters = computeEmissions(fileDir,tagCount, modifiedTrainingString)
    #save_obj(emissionParameters, fileDir, "emissionParameters")
    
    transitionParameters = computeTransitions(fileDir,tagCount, modifiedTrainingString)
    #save_obj(transitionParameters,fileDir,"transitionParameters")
    
    #use training data without tags to fit perceptrons, by using viterbi to guess the tags and then using perceptron weights to adjust
    
    #trainingSentences = []
    #with open(fileDir+"\\modifiedTrain.txt",'r',encoding='utf-8') as f:
    #    sentence = []    
    #    for line in f:
    #        if line.strip() != "":
    #            sentence.append(line.strip().split(' ')[0])
    #        else:
    #            trainingSentences.append(sentence)
    #            sentence =[]
    #    
    ##tagged data from the training set.
    #trainedSentences = []
    #with open(fileDir+"\\modifiedTrain.txt",'r',encoding='utf-8') as f:
    #    sentence = []    
    #    for line in f:
    #        if line.strip() != "":
    #            sentence.append(line.strip())
    #        else:
    #            trainedSentences.append(sentence)
    #            sentence =[]
                
                
                
    #first we change modifiedTrainingString into a list of sentences
    listOfTaggedData = modifiedTrainingString.splitlines()
    trainedSentences=[]
    sentence = []
    for line in listOfTaggedData:
        if line.strip() != "":
            sentence.append(line.strip())
        else:
            trainedSentences.append(sentence)
            sentence =[]
        
    #we do the same but strip off the tag for training perceptron
    trainingSentences = []
    sentence = []
    for line in listOfTaggedData:
        if line.strip() != "":
            sentence.append(line.strip().split(' ')[0])
        else:
            trainingSentences.append(sentence)
            sentence =[]
    
    for i in range(NUMOFITER):
        print("AT stage "+str(i)+" out of "+str(NUMOFITER))
        #For each tagged sentence in the training data, find the highest scoring tag sequence using the current weights
        numberOfSentences = len(trainingSentences)
        for sentenceIndex in range(numberOfSentences):
            viterOut = viterbiAlgorithm(trainingSentences[sentenceIndex],transitionParameters,emissionParameters).rstrip().split('\n')
            
            #If the highest scoring tag sequence matches the gold, move to next sentence
            #   If not, for each feature in the gold but not in the output, add 1 to its weight;
            #           for each feature in the output but not in the gold, take 1 from its weight
    
            trainedSentence = trainedSentences[sentenceIndex]
            sentCount = {"O":0,"B-positive":0,"I-positive":0,"B-neutral":0,"I-neutral":0,"B-negative":0,"I-negative":0}
    
            for j in range(len(trainedSentence)):   #check each word
                viterSent = viterOut[j].split(" ")[1]
                trainedSent = trainedSentence[j].split(" ")[1]
                if viterSent == trainedSent:    #correct match
                    continue
                else:   #wrong match, must change weight
                    sentCount[viterSent] -= 1
                    sentCount[trainedSent] += 1
                    #print("  mismatch found: "+viterSent+" : "+trainedSent + "  at line "+str(sentenceIndex) +" of " +str(numberOfSentences))
                
            change = True #sanity check
            #modify tagCount according to weight (in doing so, effectively change weight of viterbi. drop in tagCount will mean higher Prob)
            for senti, value in sentCount.items():
                if value != 0:
                    if tagCount[senti]-value <= 0:
                        change = False
            if change:
                for senti, value in sentCount.items():
                    if value != 0:
                        #print("  modifying with perceptron")
                        tagCount[senti]-=value
    
            emissionParameters = computeEmissions(fileDir,tagCount, modifiedTrainingString)    
            transitionParameters = computeTransitions(fileDir,tagCount, modifiedTrainingString)
    
    save_obj(transitionParameters, fileDir, "PercepTrainedTrans")
    save_obj(emissionParameters, fileDir, "PercepTrainedEmi")



#apply max-min with our new Tp Eps on "sentences". It will be dev.in for EN and FR for dev.p5.out, and also test.in for the test codes for test.p5.out
sentences = computeSentences(fileDir,modifiedTestString)
#sentences = load_obj(fileDir,"sentences")
transitionParameters = load_obj(fileDir, "PercepTrainedTrans")   
emissionParameters = load_obj(fileDir, "PercepTrainedEmi")   
part4Vit(sentences,fileDir,transitionParameters,emissionParameters)