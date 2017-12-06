# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 16:42:35 2017

@author: Sidney

Todo:
    Modularise functionality
    Modify the training set
    Write modified training set to an external function
"""

sentimentSets = ["START","STOP","O","B-positive","I-positive","B-neutral","I-neutral","B-negative","I-negative"]

#trainSetLines = trainSet.readlines()#The new lines are represented '\n'
#trainSet.close()


#Part 2.1
#tagCount is a dictionary with the tag as the key and the value being the number of occurences of the tags
def preprocess(fileDir,kVal):
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
    with open('{0}\dev.in'.format(fileDir), 'r',encoding='utf-8') as testSet:
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


#Part 2.3
def simpleSentimentAnalysis(fileDir,emissionParameters):
    #TODO: Make mind up on file.I/O
    with open('{0}\modifiedTest.txt'.format(fileDir), 'r',encoding='utf-8') as modTestSet:
        testSetString = modTestSet.read()
   
    testSetLines = testSetString.splitlines()
    devp2out = ""
    for i in testSetLines:
        if (i != ''):
            paramList = emissionParameters[i]["parameters"]
            tag = max(paramList, key=paramList.get)
            devp2out = devp2out +i +" "+tag +"\n"
        else:
            devp2out = devp2out+ '\n'    
    #TODO: Make mind up on file.I/O    
    with open('{0}\dev.p2.out'.format(fileDir), 'w',encoding='utf-8') as outputFile:
        outputFile.write(devp2out)

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

    

def main():
    languages = ["EN", "FR","SG","CN"]
    for i in languages:
        fileDir = i
        tagCount = preprocess(fileDir,3)
        save_obj(tagCount, fileDir, "tagCount")
        emissionParameters = computeEmissions(fileDir,tagCount)
        save_obj(emissionParameters, fileDir, "emissionParameters")
        sentimentAnalysis = simpleSentimentAnalysis(fileDir, emissionParameters)
        
if __name__ == "__main__":
    main()
  
    
        
    
