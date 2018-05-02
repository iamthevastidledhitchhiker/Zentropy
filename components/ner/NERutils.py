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
