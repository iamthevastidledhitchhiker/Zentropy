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
    if article[-1] != '.': article.append('.')
    sentences = []
    sentence = []
    for word in article:
        sentence.append(word)
        if word == '.':
            sentences.append(sentence)
            sentence = []
    return(sentences)

def findNamedEntities(namedArticle):
    names = []
    namedArticle[-1] = ''
    for word in namedArticle:
        try:
            names.append(word.split("__",1) )
        except Exception as e:
            print('EOF')

    organisations = []
    articleOrgs = []
    insideORG = False
    for name in names:
        if len(name) > 1 and name[1] == 'B-ORG':
            articleOrgs.append(name[0])
            insideORG = True
        elif insideORG and name[0] != '!!!':
                if len(name) >1 and name[1] == 'I-ORG':
                    articleOrgs[-1] += ' ' + name[0]
        elif name[0] == '!!!':
            organisations.append(list(set(articleOrgs)))
            articleOrgs = []
            insideORG = False
        else:
            insideORG = False

    return organisations

def writeArticle(splitArticle,openFile):
    for sentence in splitArticle:
        openFile.write(' '.join(sentence) + '\n')
    openFile.write('!!!' + '\n') #EOF
