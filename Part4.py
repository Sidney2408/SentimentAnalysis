# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 13:19:22 2017

Reference for emission and transition parameters
Emission = emissionParameters[word]["parameter"][tag]
transition = transitionParameters[prev_tag]["parameter"][current_tag]

@author: Sidney
"""
from collections import deque
import math 
from math import inf

def main(sentences, fileDir, tP, eP):
    fileString = ""
    for sentence in sentences:
        fileString = fileString+maxMarginal(sentence,tP,eP)+"\n"
    with open('{0}\\dev.p4.out'.format(fileDir), 'w',encoding='utf-8') as outputFile:
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

import pickle 
def save_obj(obj, fileDir, fileName ):
    with open('{0}\\variables\{1}.pkl'.format(fileDir,fileName),'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        
def load_obj(fileDir, fileName):
    with open('{0}\\variables\{1}.pkl'.format(fileDir,fileName), 'rb') as f:
        return pickle.load(f)

fileDir = "EN" #Specify the training sets to be used
transitionParameters = load_obj(fileDir, "transitionParameters")
#To access the transmission parameters, use transition[prev_tag][next_tag] 
emissionParameters = load_obj(fileDir,"emissionParameters")
#To access the emission parameters, use emission[word][tag]
sentences = load_obj(fileDir,"sentences")
main(sentences,fileDir,transitionParameters,emissionParameters)

print(maxMarginal(sentences[3],transitionParameters,emissionParameters))


