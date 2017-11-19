# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 19:25:59 2017

@author: Sidney
Reference for emission and transition parameters
Emission = emissionParameters[word]["parameter"][tag]
transition = transitionParameters[prev_tag]["parameter"][current_tag]
"""

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
    

def viterbiAlgorithm(sentence_array, transitionParameters, emissionParameters):
    tP = transitionParameters
    eP = emissionParameters 
    terminalValue = 0
    tagSets = {"O":0,"B-positive":0,"I-positive":0,"B-neutral":0,"I-neutral":0,"B-negative":0,"I-negative":0}
    markovTable = []
    prev_tag = "START"
    
    print("Commencing forward computation")
    
    for i in range(0,len(sentence_array)+1):
        tagSets = {"O":0,"B-positive":0,"I-positive":0,"B-neutral":0,"I-neutral":0,"B-negative":0,"I-negative":0}
        markovTable.append(tagSets)
        if i == 0:
            observation = sentence_array[i]
            print("Base case")
            for tag in markovTable[i]:
                markovTable[i][tag] = aUV(tP,prev_tag,tag)*bVxi(eP,observation,tag)
                
        elif(i == len(sentence_array)):
            print("terminal case")
            #You still have to iterate through every single dictionary 
            lastTag = "STOP"
            values = []            
            for tag in markovTable[i-1]:
                tempVal = markovTable[i-1][tag]*aUV(tP,tag,lastTag)
                values.append(tempVal)
            terminalValue = max(values)
        
        else:
            observation = sentence_array[i]
            print("entering recursive case")
            for tag in markovTable[i]:
                values = []     
                for prev_tag in markovTable[i-1]:
                    tempVal = markovTable[i-1][prev_tag]*aUV(tP,prev_tag,tag)*bVxi(eP,observation,tag)
                    values.append(tempVal)
                markovTable[i][tag] = max(values)
    print(markovTable)
    print("Commencing back trekking")
    return terminalValue

            
            
                

    
            
#viterbi(sentences, transitionParameters, emissionParameters)
        
testSentence = ['My', '#UNK#', 'year', 'old', 'loved', 'it', '.']
                
print(viterbiAlgorithm(testSentence,transitionParameters,emissionParameters))
    