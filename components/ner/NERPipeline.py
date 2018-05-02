import nltk.tokenize as tokenize
import pickle
import os

def read_pickle_one_by_one(pickle_file):
    with open(pickle_file, "rb") as t_in:
            while True:
                try:
                    yield pickle.load(t_in)
                except EOFError:
                    break

def sentenceSplitter(article):
    #Turn string into list of lists of sentences
    if article[-1] != '.': article.append('.') #always make sure article ends with full stop
    sentences = []
    sentence = []
    for word in article:
        sentence.append(word)
        if word == '.':
            sentences.append(sentence)
            sentence = []
    return(sentences)

def writeArticle(splitArticle,openFile):
    #write each sentence to new line of open file
    for sentence in splitArticle:
        openFile.write(' '.join(sentence) + '\n')

def findNamedEntities(namedArticle):
    #split article words with respective entity
    names = []
    namedArticle[-1] = ''
    for word in namedArticle:
        try:
            names.append(word.split("__",1) )
        except Exception as e:
            print('EOF')

    #extract organisations
    organisations = []
    insideORG = False
    for name in names:
        if len(name) > 1 and name[1] == 'B-ORG':
            organisations.append(name[0])
            insideORG = True
        elif insideORG:
                if len(name) >1 and name[1] == 'I-ORG':
                    organisations[-1] += ' ' + name[0]
        else:
            insideORG = False

    return list(set(organisations))


inputData   = [i for i in read_pickle_one_by_one("Data/summarizer_output.pickle")][0]

urls = inputData['urls'] #urls
stories = inputData['stories'] #full articles (untokenized)
summariesExtractive = inputData['summaries_extractive'] #extractive summaries (tokenized)
summaries = inputData['summaries_model'] #default summaries (tokenized)
summaries3Sent = inputData['summaries_3sent'] #first 3 sentences of full story (untokenized)

assert len(urls) == len(stories) == len(summariesExtractive) == len(summaries)

for story in stories:
    storyCombined = story.replace('\n', ' ')

    print('RUNNING TOKENIZER')
    storyTokenized = tokenize.word_tokenize(storyCombined)

    print('SPLITTING SENTENCES LINE BY LINE')
    split = sentenceSplitter(storyTokenized)

    inputFile = open(r'input.txt','w')
    writeArticle(split,inputFile)
    inputFile.close()

    print('RUNNING MODEL')
    os.system('python2.7 tagger-master/tagger.py --model tagger-master/models/english/ --input input.txt --output output.txt')

    with open(r'output.txt','r') as namedStory:
        namedStory=namedStory.read().replace('\n', ' ')

    print('NAMED ENTITIES:')
    orgs  = findNamedEntities(namedStory.split(' '))
    print(orgs)
