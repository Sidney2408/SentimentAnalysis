# -*- coding: utf-8 -*-
"""
Created on Sat Nov 11 15:32:52 2017

@author: Sidney
"""
with open('D:\ISTD\Term 6\Machine Learning\Project\EN\EN\\dev.in', 'r') as testSet:
    testSetString = testSet.read()
print(testSetString)
#Part 2.3
#Aim: create a dictionary in the style of {entity: sentiments}
#Or add into an array; there's the risk of overwriting an existing entity 
entity_sentiments = {}
goldSet = open('D:\ISTD\Term 6\Machine Learning\Project\EN\EN\\dev.out')
goldSetString = goldSet.readlines()
def entSentPairs (inputStringArray):
    entityString = ""
    sentimentString = ""
    for line in inputStringArray:
        data = line.split(" ")
        #Still need to think about the newline gap
        if(len(data)==2):
            word = data[0]
            tag = data[1].rstrip('\n')
            if(tag == "O" and entityString != "" and sentimentString != ""):
                if entityString in entity_sentiments.keys():
                    print("GG need to worry about repeated entities")
                else:
                    entity_sentiments[entityString.strip(" ")] = sentimentString.strip(" ")
                    entityString = ''
                    sentimentString = ''
            elif (tag != "O"):
                entityString = entityString+word+" "
                sentimentString = sentimentString +tag +" "
    return entity_sentiments

#entSentPairs(goldSetString)