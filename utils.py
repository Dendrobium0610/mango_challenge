import pickle


def createTrainHistory(keywords):
    history = {"train": {}, "valid": {}}
    for words in keywords:
        history["train"][words] = list()
        history["valid"][words] = list()
    return history


def loadTxt(filename):
    f = open(filename)
    context = list()
    for line in f:
        context.append(line.replace("\n", ""))
    return context


def saveDict(filename, data):
    file = open(filename, 'wb')
    pickle.dump(data, file)
    file.close()


def loadDict(fileName):
    with open(fileName, 'rb') as handle:
        data = pickle.load(handle)
    return data


def saveTxt(filenamesList, saveName):
    fp = open(saveName, "a")
    for name in filenamesList:
        fp.write(name + "\n")
    fp.close()
