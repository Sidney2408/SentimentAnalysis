# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 22:52:21 2017

@author: Sidney
"""

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
                #All alpha values are in their log forms 
                if(transition ==0):
                    alpha[0][current_tag] = -inf #Set to 0 
                else:
                    alpha[0][current_tag] = math.log10(transition)
        else:
            #Iterate over every tag in the current set 
            for current_tag in alpha[index]:
                runningTotal = 0
                
                #Summation over the previous set 
                for previous_tag in alpha[index-1]:
                    alpha_u_n = math.pow(10,alpha[index-1][previous_tag])
                    trans = aUV(tP,previous_tag,current_tag)
                    emit = bVxi(eP,observation, previous_tag)
                    runningTotal = runningTotal + alpha_u_n*trans*emit
                if(runningTotal ==0):
                    alpha[index][current_tag] = -inf
                else:
                    alpha[index][current_tag] = math.log10(runningTotal)
                    """
                    print("Running total for tag {} : {}".format(current_tag, math.log10(runningTotal)))                
                    print("alpha: {}".format(alpha[index]))
                    print("prevalpha: {}".format(alpha[index-1]))
          
        print("\n")
        print("alpha stage {0}".format(index))
        print(alpha)
        print("\n")
        """

        
    beta = deque()
    lastTag = "STOP"
    for index in range(n-1,-1,-1):
        observation = sentence[index]
        tagSets = {"O":0,"B-positive":0,"I-positive":0,"B-neutral":0,"I-neutral":0,"B-negative":0,"I-negative":0}#Forgot to refresh the array
        beta.appendleft(tagSets)
        
        
        if (index == n-1):
            #base case 
            for current_tag in beta[0]:
                transition = aUV(tP,current_tag,lastTag)
                emission = bVxi(eP,observation,current_tag)
                beta_u_n = transition*emission
                if(beta_u_n ==0):
                    beta[0][current_tag] = -inf
                else:
                    beta[0][current_tag] = math.log10(beta_u_n)
                    
        else:
            prevTagSet = beta[1]
            for current_tag in beta[0]:
                runningTotal = 0
                for previous_tag in prevTagSet:
                    beta_u_n = math.pow(10,prevTagSet[previous_tag])
                    trans = aUV(tP,current_tag,previous_tag)#Reversed flow 
                    emit = bVxi(eP,observation, previous_tag)
                    runningTotal += beta_u_n*trans*emit
                
                if(runningTotal ==0):
                    beta[0][current_tag] = -inf
                else:
                    beta[0][current_tag] = math.log10(runningTotal)
        
        print("beta stage {0}".format(index))
        print(beta)
        print("\n")
        
        
    obs_statePair = ""   
    for index in range(0,n):
        word = sentence[index]
        dict_alpha = alpha[index]
        dict_beta =  beta[index]
        #alpha_x_beta = {k:dict_alpha[k]*dict_beta[k] for k in tagSets}
        alpha_x_beta = {}
        for sentiment in tagSets:
            alpha_x_beta[sentiment] = dict_alpha[sentiment]*dict_beta[sentiment]
        tag = min(alpha_x_beta, key=alpha_x_beta.get)
        #print("word: {}, tag: {}".format(word,tag))
        obs_statePair = obs_statePair +word +" "+tag +"\n"
    return obs_statePair