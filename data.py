import nltk
import os, sys, email,re
import random
import numpy as np
import pandas as pd
from nltk.corpus import stopwords 
from nltk.stem.wordnet import WordNetLemmatizer
import string
from nltk.stem.porter import PorterStemmer
from nltk.tokenize.regexp import RegexpTokenizer

from subprocess import check_output

from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer

def email_from_string(raw_email):
    """
    Common function from notebooks on kaggle to extract emails
    """
    msg = email.message_from_string(raw_email)
    
    content = []
    for part in msg.walk():
        if part.get_content_type() == 'text/plain':
            content.append(part.get_payload())
            
    result = {}
    for key in msg.keys(): 
        result[key] = msg[key]
    result["content"] = ''.join(content)
    
    return result

def remove_fwd_org(test_email):
    """
    this function clears out fowarded and original messages from e-mail along with other junk.
    """
    lines  = test_email.split("\n")
    i = 0
    Fowarded = "-- Forwarded"
    Original = "-----Original Message-----"
    N = len(lines)
    while i < N:
        line  = lines[i]
        if Fowarded in line:
            lines = lines[:i]
            i = N
        if Original in line:
            lines = lines[:i]
            i = N
        if "Sent by:" in line:
            lines = lines[:i]
            i = N
        if "From:" in line:
            lines = lines[:i]
            i = N
        if "Subject:" in line:
            lines = lines[:i-7]
            i = N
        if "To:" in line:
            lines = lines[:i-3]
            i = N
        i+=1

    lines  = [line+"\n" for line in lines]

    new_email = "".join(lines).strip("\n")
    return new_email

def before(value, a):
    """
    Find first part and return slice before it.
    """
    pos_a = value.find(a)
    if pos_a == -1: return ""
    return value[0:pos_a]


def get_data():
    """
    Cleans up the content of emails and extracts what times emails are sent
    """
    enron_data = pd.read_csv("emails.csv", header=0, quoting=2)
    enron_sent = enron_data[enron_data["file"].str.contains('sent').tolist()]
    enron_sent['start'], enron_sent['end'], enron_sent['fileno'] = zip(*enron_sent['file'].map(lambda x: x.split('/')))
    enron_sent = enron_sent.assign(sender=enron_sent["file"].map(lambda x: re.search("(.*)/.*sent", x).group(1)).values)
    enron_sent.drop("file", axis=1, inplace=True)
    enron_parsed = pd.DataFrame(list(map(email_from_string, enron_sent.message)))
    df = enron_parsed[enron_parsed.From.str.contains("enron")]
    df = df[:][~pd.isnull(df.To)]
    df = df[df.To.str.contains("enron")]
    df = df[:][pd.isnull(df.Bcc)]
    df = df[:][pd.isnull(df.Cc)]
    df = df[:][df.To.apply(lambda x: len(str(x).split(","))) == 1]
    df.content = df.content.str.strip("\n")
    df = df[:][~df.content.str.split("\n").apply(lambda x : "-- Forwarded" in x[0])]
    df = df[:][~df.content.str.split("\n").apply(lambda x : "-- Inline" in x[0])]
    df["content"] = df.content.apply(remove_fwd_org)
    df = df[:][~(df.content == "")]
    df["Date"] =  pd.to_datetime(df["Date"], infer_datetime_format=True)
    df["Time"] = df["Date"].dt.hour
    return df

def get_sender_data(save_file = False):
    """
    Assigns gender to senders based on their first name.
    Can return a csv if save_file = True
    """
    df = get_data
    names = []
    for i in df.From:
        names.append(before(i,'@'))

    names = pd.DataFrame(names)
    
    names.columns = ["fullname"]
    first_names = []
    for i in names.fullname:
        first_names.append(before(i,'.')) 
        
    first_names = pd.DataFrame(first_names)
    first_names['gender']=np.zeros(len(first_names))
    first_names.columns = ['name','gender']
    
    male_list = [
        '','a','alhamd','andrea', 'b', 'c',
        'casey', 'click',  'd', 'dana',
        'dutch','e', 'enron','f',  'gaby', 'geir', 'gretel', 'h','houston <', 'hunter', 'j', 
        'joe','k', 'kam',  'kay', 'kaye', 'kim',  
        'l', 'larry', 'legal <', 'lindy', 
        'lynn', 'lysa', 'm', 'madhup','no','pinnamaneni',  'robin', 's','stacy', 't', 
        'teb', 'terry','tori','tracy', 'trading <', 'twanda',  'v', 'vasant','vladi',
        ]
    
    first_names.gender[first_names.name.isin(male_list)]='I'

    first_names.gender[first_names.name.isin([
        'phillip','albert','andrew','benjamin','bert','bill','brad','daren','darrell',
        'darron', 'david','tom','thomas','vince','paul','peter','larry','kevin','kenneth',
        'keith','juan','mike','louise','mike','monika',
        'joseph','jonathan','john','joe','jim','jeffrey','jeff','jason','james','hunter',
        'harry','cooper','stinson','stanley','robert','rogers','scott', 'scotty', 'sean',
        'rod','chris','eric','mark','martin','matt', 'matthew',  'michael','carl','clint',  
        'craig','charles','geoff','vince','paul','randall','peter','phillip','torrey','rob',
        'holden','jay','doug','don','harry','greg','grant','gerald','frank',
        'fletcher','danny','darren','barry','andy','errol','dan'
    
    ])]='M'

    first_names.gender[first_names.name.isin([
        'amelia','angela', 'brenda','carol','ursula','theresa','susan', 'suzanne', 'sylvia', 
        'stephanie','steven','sonia','sara','rosalee', 'ryan','richard', 'rick', 
        'sheila', 'shelley','kimberly','margaret', 'marie','sandra','maureen',
        'melissa','cindy', 'jenny', 'martha','mary','michelle','lisa', 'liz','shirley',
        'monique','tori','sally', 'tamara','tana','sherri','patrice','patti',
        'paige','pam','judy','diana','kate','katherine', 'katrina','gwyn',
        'elizabeth','debra','cheryl','cathy','cara','jane','janette','ina','drew'
    
    ])]='F'
    sender_data = pd.concat([first_names,df.content,df.Time],axis=1)
    if save_file:
        sender_data.to_csv('sender_data.csv')
    else:
        return sender_data

def get_recipient_data(save_file = False):
    """
    Assigns gender to recipients based on their first name.
    Can return a csv if save_file = True
    """
    names = []
    for i in df.To:
        names.append(before(i,'@'))

    names = pd.DataFrame(names)    
    names.columns = ["fullname"]
    first_names = []
    for i in names.fullname:
        first_names.append(before(i,'.'))
        
    first_names = pd.DataFrame(first_names)
    first_names['gender']=np.zeros(len(first_names))
    first_names.columns = ['name','gender']
    names_database = pd.read_csv('name_gender.csv')
    names_database = names_database[names_database.probability > .95]
    F_labels = names_database[names_database.gender=='F']
    M_labels = names_database[names_database.gender=='M']
    F_labels.name = F_labels.name.str.lower()
    M_labels.name = M_labels.name.str.lower()
    first_names.gender[first_names.name.isin(F_labels.name)]='F'
    first_names.gender[first_names.name.isin(M_labels.name)]='M'
    first_names.gender[first_names.gender==0]='I'
    recipient_data = pd.concat([first_names,df.content,df.Time],axis=1)
    
    if save_file:
        recipient_data.to_csv('recipient_data.csv')
    else:
        return recipient_data
    