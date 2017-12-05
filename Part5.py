# -*- coding: utf-8 -*-

from collections import deque
from collections import defaultdict


def part4Vit(sentences, fileDir, tP, eP):
    fileString = ""
    for sentence in sentences:
        maxMarginal(sentence,tP,eP)
        fileString = fileString+maxMarginal(sentence,tP,eP)+"\n"
        
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





### PART 2

sentimentSets = ["START","STOP","O","B-positive","I-positive","B-neutral","I-neutral","B-negative","I-negative"]
def preprocess(fileDir,kVal):
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
    with open('{0}\test.in'.format(fileDir), 'r',encoding='utf-8') as testSet:
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
    
    with open('{0}\modifiedTrain.txt'.format(fileDir), 'w',encoding='utf-8') as outputFile:
        outputFile.write(modifiedString)
        
    with open('{0}\modifiedTest.txt'.format(fileDir), 'w',encoding='utf-8') as outputTestFile:
        outputTestFile.write(modifiedTestString)        
    return tagCount

def dictProcess(dictionary, key):
    dictionary[key] = dictionary.get(key,0)+1 


#Returns a dictionary with the emission parameters  
def computeEmissions(fileDir, tagCount):
    with open('{0}\modifiedTrain.txt'.format(fileDir), 'r',encoding='utf-8') as modTrainSet:
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


import pickle 
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

from Part3 import computeTransitions,computeSentences, viterbiAlgorithm


fileDir = "FR"
#Part 2 cleans train and test data. Test data is further parsed in part 3, into 'sentences'
#Otherwise, test data ('sentences') is not touched, and train data is used to generate emis and trans parameters
#part 3 also uses viterbi to train the params. We modify this to train perceptrons

tagCount = preprocess(fileDir,3)
save_obj(tagCount, fileDir, "tagCount")
emissionParameters = computeEmissions(fileDir,tagCount)
save_obj(emissionParameters, fileDir, "emissionParameters")

transitionParameters = computeTransitions(fileDir,tagCount)
save_obj(transitionParameters,fileDir,"transitionParameters")


trainingSentences = []
with open(fileDir+"\\modifiedTrain.txt",'r',encoding='utf-8') as f:
    sentence = []    
    for line in f:
        if line.strip() != "":
            sentence.append(line.strip().split(' ')[0])
        else:
            trainingSentences.append(sentence)
            sentence =[]
    


trainedSentences = []
with open(fileDir+"\\modifiedTrain.txt",'r',encoding='utf-8') as f:
    sentence = []    
    for line in f:
        if line.strip() != "":
            sentence.append(line.strip())
        else:
            trainedSentences.append(sentence)
            sentence =[]
    
NUMOFITER = 10
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
                print("  mismatch found: "+viterSent+" : "+trainedSent + "  at line "+str(sentenceIndex) +" of " +str(numberOfSentences))
            
        change = True #sanity check
        #modify tagCount according to weight (in doing so, effectively change weight of viterbi. drop in tagCount will mean higher Prob)
        for senti, value in sentCount.items():
            if value != 0:
                if tagCount[senti]-value <= 0:
                    change = False
        if change:
            for senti, value in sentCount.items():
                if value != 0:
                    print("  modifying with perceptron")
                    tagCount[senti]-=value

        #save_obj(tagCount, fileDir, "tagCount")
        emissionParameters = computeEmissions(fileDir,tagCount)
        #save_obj(emissionParameters, fileDir, "emissionParameters")
        
        transitionParameters = computeTransitions(fileDir,tagCount)
        #save_obj(transitionParameters,fileDir,"transitionParameters")


        #print(str(tagCount)+'\n')
        #Return weights

#apply max-min with our new Tp Eps
        
computeSentences(fileDir)
sentences = load_obj(fileDir,"sentences")

part4Vit(sentences,fileDir,transitionParameters,emissionParameters)


