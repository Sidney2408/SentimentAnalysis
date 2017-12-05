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



from Part2 import preprocess,computeEmissions, save_obj
from Part3 import computeTransitions,computeSentences, load_obj, viterbiAlgorithm


fileDir = "EN"
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
    
NUMOFITER = 3
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
        #emissionParameters = computeEmissions(fileDir,tagCount)
        #save_obj(emissionParameters, fileDir, "emissionParameters")
        
        transitionParameters = computeTransitions(fileDir,tagCount)
        #save_obj(transitionParameters,fileDir,"transitionParameters")


        #print(str(tagCount)+'\n')
        #Return weights

#apply max-min with our new Tp Eps
        
computeSentences(fileDir)
sentences = load_obj(fileDir,"sentences")

part4Vit(sentences,fileDir,transitionParameters,emissionParameters)


