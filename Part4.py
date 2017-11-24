# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 13:19:22 2017

@author: Sidney
"""
from collections import deque


fileDir = "FR" #Specify the training sets to be used
transitionParameters = load_obj(fileDir, "transitionParameters")
#To access the transmission parameters, use transition[prev_tag][next_tag] 
emissionParameters = load_obj(fileDir,"emissionParameters")
#To access the emission parameters, use emission[word][tag]
sentences = load_obj(fileDir,"sentences")

def main(sentences, fileDir, tP, eP):
    fileString = ""
    for sentence in sentences:
        fileString = fileString+maxMarginal(sentence,tP,eP)+"\n"
    with open('{0}\\dev.p3.out'.format(fileDir), 'w',encoding='utf-8') as outputFile:
        outputFile.write(fileString)
        
def maxMarginal(sentence,tP,eP):
    n = len(sentence)
    alpha = deque()
    firstTag = "START"
    for i in range(0,n):
        tagSets = {"O":0,"B-positive":0,"I-positive":0,"B-neutral":0,"I-neutral":0,"B-negative":0,"I-negative":0}
        alpha.append(tagSets)
        if (i == 0):
            #Base case
            alpha_u_j = 0
            for current_tag in alpha[i]:
                transition = aUV(tP,firstTag,current_tag)
                if(transition ==0):
                    pass#Ignore the current value 
                else:
                    alpha[0][current_tag] = math.log10(transition)
                    
                
            
                
        
    beta = deque()
    lastTag = "STOP"
    for i in range(0,n):
        tagSets = {"O":0,"B-positive":0,"I-positive":0,"B-neutral":0,"I-neutral":0,"B-negative":0,"I-negative":0}
        
def aUV(transitionParameters,prev_tag,tag):
    dic = transitionParameters[prev_tag]["parameters"]
    return dic.get(tag,0)
     
def bVxi(emissionParameters,observation,tag):
    dic= emissionParameters[observation]["parameters"]
    return dic.get(tag,0)