import nltk.tokenize as tokenize
import pickle
import os
import NERutils as neru

inputData   = [i for i in neru.read_pickle_one_by_one("Data/summarizer_output.pickle")][0]

urls = inputData['urls']
stories = inputData['stories']
summariesExtractive = inputData['summaries_extractive']
summaries = inputData['summaries_model']

assert len(urls) == len(stories) == len(summariesExtractive) == len(summaries)

inputFile = open(r'input.txt','w')

for story in stories:
    storyCombined = story.replace('\n', ' ')

    print('RUNNING TOKENIZER')
    storyTokenized = tokenize.word_tokenize(storyCombined)

    print('SPLITTING SENTENCES LINE BY LINE')
    split = neru.sentenceSplitter(storyTokenized)

    neru.writeArticle(split,inputFile)

inputFile.close()

print('RUNNING MODEL')
os.system('python2.7 tagger-master/tagger.py --model tagger-master/models/english/ --input input.txt --output output.txt')

with open(r'output.txt','r') as namedStory:
    namedStory=namedStory.read().replace('\n', ' ')

print('NAMED ENTITIES:')
orgs  = neru.findNamedEntities(namedStory.split(' '))
print(orgs)
