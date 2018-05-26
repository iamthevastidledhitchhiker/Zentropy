import nltk.tokenize as tokenize
import pickle
import os
import csv
import NERutils as neru

inputData   = [i for i in neru.read_pickle_one_by_one("Data/summarizer_output.pickle")][0]


reader = csv.DictReader(open('Data/test_roman.csv'))
labels = {}
for row in reader:
    key = row.pop('ids')
    labels[key] = row.pop('aspect_name')

reader = csv.DictReader(open('Data/train_roman.csv'))
for row in reader:
    key = row.pop('ids')
    labels[key] = row.pop('aspect_name')

# No = 3
# print('ID:')
# print(inputData['ids'][No])
# print('Full Story:')
# print(inputData['stories'][No])
# print('3 Sentence Summary:')
# print(inputData['summaries_3sent'][No])
# print('Aspect Term:')
# print(labels[inputData['ids'][No]])


print(inputData.keys())

#3 SENTENCE SUMMARIES
inputFile = open(r'input.txt','w')
for story in inputData['summaries_3sent']:
    storyCombined = story.replace('\n', ' ')
    storyTokenized = tokenize.word_tokenize(storyCombined)
    split = neru.sentenceSplitter(storyTokenized)
    neru.writeArticle(split,inputFile)
inputFile.close()

print('RUNNING MODEL')
os.system('python2.7 tagger-master/tagger.py --model tagger-master/models/english/ --input input.txt --output output.txt')

with open(r'output.txt','r') as namedStory:
    namedStory=namedStory.read().replace('\n', ' ')

orgs  = neru.findNamedEntities(namedStory.split(' '))

neru.writeOutput(orgs,inputData['summaries_3sent'],'Aspect_3Sent')

N = len(inputData['ids'])

score = 0
for j in range(N):
    for articleOrg in orgs[j]:
        if articleOrg == labels[inputData['ids'][j]]:
            score += 1#/(len(orgs[j]))
print('3 Sentence Summary F1: ' + str(score/N))

#No COVERAGE
inputFile = open(r'input.txt','w')
for story in inputData['summaries_no_coverage']:
    storyCombined = story.replace('\n', ' ')
    storyTokenized = tokenize.word_tokenize(storyCombined)
    split = neru.sentenceSplitter(storyTokenized)
    neru.writeArticle(split,inputFile)
inputFile.close()

print('RUNNING MODEL')
os.system('python2.7 tagger-master/tagger.py --model tagger-master/models/english/ --input input.txt --output output.txt')

with open(r'output.txt','r') as namedStory:
    namedStory=namedStory.read().replace('\n', ' ')

orgs  = neru.findNamedEntities(namedStory.split(' '))

neru.writeOutput(orgs,inputData['summaries_no_coverage'],'Aspect_NoCoverage')

N = len(inputData['ids'])

score = 0
for j in range(N):
    for articleOrg in orgs[j]:
        if articleOrg == labels[inputData['ids'][j]]:
            score += 1#/(len(orgs[j]))
print('No Coverage Summary F1: ' + str(score/N))

#SOME COVERAGE
inputFile = open(r'input.txt','w')
for story in inputData['summaries_some_coverage']:
    storyCombined = story.replace('\n', ' ')
    storyTokenized = tokenize.word_tokenize(storyCombined)
    split = neru.sentenceSplitter(storyTokenized)
    neru.writeArticle(split,inputFile)
inputFile.close()

print('RUNNING MODEL')
os.system('python2.7 tagger-master/tagger.py --model tagger-master/models/english/ --input input.txt --output output.txt')

with open(r'output.txt','r') as namedStory:
    namedStory=namedStory.read().replace('\n', ' ')

orgs  = neru.findNamedEntities(namedStory.split(' '))

neru.writeOutput(orgs,inputData['summaries_some_coverage'],'Aspect_SomeCoverage')

N = len(inputData['ids'])

score = 0
for j in range(N):
    for articleOrg in orgs[j]:
        if articleOrg == labels[inputData['ids'][j]]:
            score += 1#/(len(orgs[j]))
print('Some Coverage Summary F1: ' + str(score/N))

#MORE COVERAGE
inputFile = open(r'input.txt','w')
for story in inputData['summaries_more_coverage']:
    storyCombined = story.replace('\n', ' ')
    storyTokenized = tokenize.word_tokenize(storyCombined)
    split = neru.sentenceSplitter(storyTokenized)
    neru.writeArticle(split,inputFile)
inputFile.close()

print('RUNNING MODEL')
os.system('python2.7 tagger-master/tagger.py --model tagger-master/models/english/ --input input.txt --output output.txt')

with open(r'output.txt','r') as namedStory:
    namedStory=namedStory.read().replace('\n', ' ')

orgs  = neru.findNamedEntities(namedStory.split(' '))

neru.writeOutput(orgs,inputData['summaries_more_coverage'],'Aspect_MoreCoverage')

N = len(inputData['ids'])

score = 0
for j in range(N):
    for articleOrg in orgs[j]:
        if articleOrg == labels[inputData['ids'][j]]:
            score += 1#/(len(orgs[j]))
print('More Coverage Summary F1: ' + str(score/N))
