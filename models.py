import nltk
import os, sys, email,re
import numpy as np
import pandas as pd
from nltk.corpus import stopwords 
from nltk.stem.wordnet import WordNetLemmatizer
import string
from nltk.stem.porter import PorterStemmer
from nltk.tokenize.regexp import RegexpTokenizer
from dateutil.parser import parse
from subprocess import check_output
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression


#########################################################
#################### Preprocessing ######################
#########################################################


eng_stopwords = set(stopwords.words('english'))
def clean_text(text):
    """
    This Function was taken from one of the notebooks on kaggle where we got the Enron Data Set
    https://www.kaggle.com/jaykrishna/topic-modeling-enron-email-dataset.
    From looking it over this function takes some text and 
    """
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.lower().split()
    words = [w for w in words if w not in eng_stopwords]
    return ' '.join(words)

def is_date(string):
    try: 
        parse(string)
    except ValueError:
        return False
    except OverflowError:
        return False
    else:
        return True

def is_float(string):
    try:
        float("".join(string.split(",")))
    except ValueError:
        return False
    except OverflowError:
        return True
    else:
        return True

def replace_nums(text):
    words=text.split(" ")
    for i in range(len(words)):
        word = words[i]
        word_date = is_date(word)
        word_float = is_float(word)
        if word_date:
            words[i] = "/date/"
        if word_float:
            words[i] = "/number/"
    else:
        words = [word + " " for word in words]
        new_text = "".join(words).strip(" ")
    return new_text

def bow_vectorizer(text, vector, max_len):
    words = text.split(" ")
    N =  len(words) - 2 + 1
    bows = [words[i]+" "+words[i+1] for i in range(N)]
    L = vector.transform(bows)
    return L

    
def trouble_arrows(text):
    if "> >" in text:
        text = text.replace(">", "").strip(" ")
    return text    

def preprocess(filename):
    """
    Does preprocessing on dataset retrrieved from filename
    """
    df = pd.read_csv(filename)
    df = df[:][~df.content.isnull()]
    df = df[:][~df.gender.isnull()]
    df = df[:][~(df.gender == "0.0")]
    df = df[:][~(df.gender == "I")]
    df.content = df.content.str.replace('\n'," ")
    df.content = df.content.apply(replace_nums)
    df.content = df.content.apply(trouble_arrows)
    df = df[:][~(df.content.apply(lambda x: len(x.split(" ")))<=3)]
    df = df[:][~(df.content.apply(lambda x: len(x.split(" ")))>=201)]
    df["clean_content"] = df.content.apply(clean)
    vectorizer = CountVectorizer().fit(df.content)
    max_len = int(df.content.apply(lambda x: len(x.split(" "))).max())
    df["bow_vectors"] = df.content.apply(lambda x: bow_vectorizer(x, vectorizer, max_len))
    df["label"] = df.gender.apply(lambda x: torch.LongTensor([1*(x=="F")]))
    df["max_len"] = max_len
    return df

def get_splits(df):
    train_indices, other_indices = train_test_split(np.arange(len(df)), test_size=0.30)
    val_indices, test_indices = train_test_split(other_indices, test_size = 0.6667)
    return train_indices, val_indices, test_indices

def split_dataframe(df, splits):
    train_indices, val_indices, test_indices = splits
    training_set = df.iloc[train_indices]
    dev_set = df.iloc[val_indices]
    test_set = df.iloc[test_indices]
    return training_set, dev_set, test_set



def fasttext_formatter(df, file_name):
    f = open(file_name, "w")
    for i in range(len(df)):
        current = df.iloc[i]
        if current.gender == "M":
            label = "__label__Male "
        else:
            label = "__label__Female "
        line =  label + current.content + "\n"
        f.write(line)
    else:
        f.close()
        
#########################################################
###################### SVM ##############################
#########################################################


def SVM_regulizer(X_train, y_train, X_val, y_val, Cs):
    """
    Runs multiple SVMs at different values of C(reciprocal of regulariazation) and returns a list of accuracies
    """
    accs = []
    for c in Cs:
        model = LinearSVC(C = c)
        model = model.fit(X_train, y_train)
        pred_y = model.predict(X_val)
        acc = sum(y_val.values == pred_y)/len(pred_y)
        accs.append(acc)
    return accs

def SVM_optimizer(df, splits, powers):
    splits = get_splits(df)
    train_indices, val_indices, test_indices = splits
    Cs = 10.0**powers
    wordvector = TfidfVectorizer(analyzer='word', stop_words='english', max_df=0.4, min_df=5)
    wordvector_fit = wordvector.fit_transform(df.clean_content)
    X_train, X_val, X_test = wordvector_fit[train_indices], wordvector_fit[val_indices], wordvector_fit[test_indices]
    y_train, y_val, y_test = split_dataframe(df.gender, splits)
    accs = SVM_regulizer(X_train, y_train, X_val, y_val, Cs)
    max_power = powers[np.argmax(np.array(accs))]
    model = LinearSVC(C = 10.0**max_power)
    model = model.fit(X_train, y_train)
    best_y = model.predict(X_test)
    test_acc = sum(y_test.values == best_y)/len(best_y)
    print("The accuracy of the final SVM model is %s" % (test_acc))
    return test_acc, 10.0**max_power

def logreg_regulizer(X_train, y_train, X_val, y_val, Cs):
    """
    Runs multiple Logiscti Regressions at different values of C(reciprocal of regulariazation) and returns a list of accuracies
    """
    accs = []
    for c in Cs:
        model = LogisticRegression(C = c)
        model = model.fit(X_train, y_train)
        pred_y = model.predict(X_val)
        acc = sum(y_val.values == pred_y)/len(pred_y)
        accs.append(acc)
    return accs

def logreg_optimizer(df, splits, powers):
    splits = get_splits(df)
    train_indices, val_indices, test_indices = splits
    Cs = 10.0**powers
    wordvector = TfidfVectorizer(analyzer='word', stop_words='english', max_df=0.4, min_df=5)
    wordvector_fit = wordvector.fit_transform(df.clean_content)
    X_train, X_val, X_test = wordvector_fit[train_indices], wordvector_fit[val_indices], wordvector_fit[test_indices]
    y_train, y_val, y_test = split_dataframe(df.gender, splits)
    accs = logreg_regulizer(X_train, y_train, X_val, y_val, Cs)
    max_power = powers[np.argmax(np.array(accs))]
    model = LinearSVC(C = 10.0**max_power)
    model = model.fit(X_train, y_train)
    best_y = model.predict(X_test)
    test_acc = sum(y_test.values == best_y)/len(best_y)
    print("The accuracy of the final Logistic Regression model is %s" % (test_acc))
    return test_acc, 10.0**max_power


def ADAboost_lr(X_train, y_train, X_val, y_val, LRs):
    
    for lr in LRs:
        model = AdaBoostClassifier(learning_rate = lr)
        model = model.fit(X_train, y_train)
        pred_y = model.predict(X_val)
        acc = sum(y_val.values == pred_y)/len(pred_y)
        accs.append(acc)
    return accs
    
    
def ADAboost_optimizer(df, splits, powers):
    splits = get_splits(df)
    train_indices, val_indices, test_indices = splits
    lrs = 10.0**powers
    wordvector = TfidfVectorizer(analyzer='word', stop_words='english', max_df=0.4, min_df=5)
    wordvector_fit = wordvector.fit_transform(df.clean_content)
    X_train, X_val, X_test = wordvector_fit[train_indices], wordvector_fit[val_indices], wordvector_fit[test_indices]
    y_train, y_val, y_test = split_dataframe(df.gender, splits)
    accs = ADAboost_lr(X_train, y_train, X_val, y_val, lrs)
    max_power = powers[np.argmax(np.array(accs))]
    model = AdaBoostClassifier(learning_rate = 10.0**max_power)
    model = model.fit(X_train, y_train)
    best_y = model.predict(X_test)
    test_acc = sum(y_test.values == best_y)/len(best_y)
    print("The accuracy of the final Ada Boost model is %s" % (test_acc))
    return test_acc, 10.0**max_power




#########################################################
################### bow-CNN #############################
#########################################################

### NOTE: A decent amount of this code is a copy of/based off the code provided in HW 3


def padder(L, max_len):
    L = L.todense(out=np.zeros(L.shape, L.dtype))
    L = torch.FloatTensor(L.tolist())
    if L.shape[0] < max_len:
        diff = max_len - L.shape[0]
        V = L.shape[1]
        padding = torch.FloatTensor(diff, V).zero_()
        
        
        L = torch.cat([L, padding], 0)
    return L

# This is the iterator we'll use during training. 
# It's a generator that gives you one batch at a time.
def data_iter(source, batch_size):
    dataset_size = len(source)
    start = -1 * batch_size
    order = list(range(dataset_size))
    random.shuffle(order)

    while True:
        start += batch_size
        if start > dataset_size - batch_size:
            # Start another epoch.
            start = 0
            random.shuffle(order)   
        batch_indices = order[start:start + batch_size]
        yield [source.iloc[index] for index in batch_indices]

# This is the iterator we use when we're evaluating our model. 
# It gives a list of batches that you can then iterate through.
def eval_iter(source, batch_size):
    batches = []
    dataset_size = len(source)
    start = -1 * batch_size
    order = list(range(dataset_size))
    random.shuffle(order)

    while start < dataset_size - batch_size:
        start += batch_size
        batch_indices = order[start:start + batch_size]
        batch = [source.iloc[index] for index in batch_indices]
        batches.append(batch)
        
    return batches

# The following function gives batches of vectors and labels, 
# these are the inputs to your model and loss function
def get_batch(batch):
    vectors = []
    labels = []
    for series in batch:
        max_len = series["max_len"]
        vectors.append(padder(series["bow_vectors"], max_len))
        labels.append(series["label"])
    return vectors, labels

def evaluate(model, data_iter):
    model.eval()
    correct = 0
    total = 0
    for i in range(len(data_iter)):
        vectors, labels = get_batch(data_iter[i])
        vectors = Variable(torch.stack(vectors).squeeze())
        labels = torch.stack(labels).squeeze()
        output = model(vectors)
        _, predicted = torch.max(output.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
    return correct / float(total)

def training_loop(model, loss, optimizer, training_iter, num_train_steps,
                  dev_iter = [], train_eval_iter = [],  verbose = True):
    step = 0
    for i in range(num_train_steps):
        model.train()
        vectors, labels = get_batch(next(training_iter))
        vectors = Variable(torch.stack(vectors).squeeze())
        labels = Variable(torch.stack(labels).squeeze())

        model.zero_grad()
        output = model(vectors)

        lossy = loss(output, labels)
        lossy.backward()
        optimizer.step()
        if verbose == False:
            pass
        elif step % 50 == 0:
            print( "Step %i; Loss %f; Train acc: %f; Dev acc %f" 
                %(step, lossy.data[0], evaluate(model, train_eval_iter), evaluate(model, dev_iter)))

        step += 1
        
class bowCNN(nn.Module):
    def __init__(self, vocab_size, window_size, n_filters, num_labels):
        super(bowCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, n_filters, (window_size, vocab_size))
        self.fc1 = nn.Linear(n_filters, num_labels)
        self.init_weights()
        
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv1(x).squeeze(3)
        x = F.relu(x)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        x = self.fc1(x)
        return x
    
    def init_weights(self):
        initrange = 0.1
        lin_layers = [self.fc1]
     
        for layer in lin_layers:
            layer.weight.data.uniform_(-initrange, initrange)
            layer.bias.data.fill_(0)
    
    
def CNN_trainer(training_iter, num_train_steps, vocab_size, window_size, n_filters, lr):
    model = bowCNN(vocab_size, window_size, n_filters, num_labels)
    
    # Loss and Optimizer
    loss = nn.CrossEntropyLoss()  
    optimizer = torch.optim.Adam(model.parameters(), lr)

    # Train the model
    training_loop(model, loss, optimizer, training_iter, num_train_steps)
    return model

    
    
    