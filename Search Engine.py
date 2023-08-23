import os
from nltk.tokenize import RegexpTokenizer
import nltk
import math
from collections import Counter
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

processed_collection = {}
term_frequency_for_each_doc = {}
#nltk.download()

## This function reads all the documents and returns a dictinoary{docname,List of tokenized and stemmed words}.
def preprocess():
    corpusroot = './US_Inaugural_Addresses'
    all_files = os.listdir(corpusroot)
    valid_files = list(filter(lambda x: x.endswith(".txt"), all_files))
    collection = {}
    for each_file in valid_files:
        file = open(corpusroot + "/" + each_file, "r")
        content = file.read()
        file.close()
        content = content.lower()
        processed_doc = tokenizer_stemmer(content)
        collection[each_file] = processed_doc
    return collection

## This function takes single document and tokenize and apply stemmer
def tokenizer_stemmer(document):
    tokenizer = RegexpTokenizer(r'[a-zA-Z]+')
    tokens = tokenizer.tokenize(document)
    stop_words = stopwords.words('english')
    processed_doc = []
    for token in tokens:
        if token not in stop_words:
            final_token = apply_stemmer(token)
            processed_doc.append(final_token)
    return processed_doc

## This function Param {docname,list of tokenized ans stemmed words}
## returns {docname,Counter{tokenized and stemword,frequency}}
def get_term_frequency(collection):
    term_frequency = {}
    for key in collection.keys():
        cnt = Counter(collection[key])
        term_frequency[key] = cnt
    return term_frequency

## This function takes param token and returns idf: If tf is zero return -1
def getidf(token):
    final_token = apply_stemmer(token)
    dft = 0
    for each_file in term_frequency_for_each_doc.keys():
        counter = term_frequency_for_each_doc[each_file]
        if counter[final_token] > 0:
            dft = dft + 1
    if dft == 0:
        return -1
    return math.log10(len(processed_collection.keys()) / dft)

## Lib function to apply stemming on the token
def apply_stemmer(token):
    stemmer = PorterStemmer()
    final_token = stemmer.stem(token)
    return final_token

## Function takes filename,token returns normalized tf-idf: if tf is zero return 0
def getweight(filename, token):
    idf = getidf(token)
    final_token = apply_stemmer(token)
    term_frequency_dic = term_frequency_for_each_doc[filename]
    if term_frequency_dic[final_token] == 0:
        return 0
    magnitude = 0
    for each_token in term_frequency_dic.keys():
        idf_current = getidf(each_token)
        tf = term_frequency_dic[each_token]
        weight = (1+ math.log10(tf)) * idf_current
        magnitude = magnitude+ (weight**2)
    magnitude = math.sqrt(magnitude)
    return ((1+math.log10(term_frequency_dic[final_token]))*idf)/magnitude

## Function takes query and apply ltc.lnc returns the
## topmatched document name and socre.
## if there is no match the doc with zero score is answer
def query(query):
    ######## Query lnc calculation ############
    toke_query = apply_stemmer(query)
    query_doc = toke_query.split()
    query_counter_dict = Counter(query_doc)

    query_log_tf_dict ={}
    maginitude = 0
    for each_token in query_counter_dict.keys():
        tf = query_counter_dict[each_token]
        query_log_tf_dict[each_token] = 1+math.log10(tf)
        maginitude = maginitude+ ((1+math.log10(tf))**2)
    nor_maginitude = math.sqrt(maginitude)
    query_cosine_dict = {}
    for each_token in query_log_tf_dict.keys():
        tf = query_log_tf_dict[each_token]
        query_cosine_dict[each_token] = tf/nor_maginitude
    ######## End Query lnc calculation ############

    ######## Document ltc calculation ############
    high_score = 0
    document_ans = "No Match"
    for each_documet in term_frequency_for_each_doc.keys():
        document_counter_dict = term_frequency_for_each_doc[each_documet]
        document_weight_tf_dict = {}
        maginitude = 0
        for each_token in document_counter_dict.keys():
            tf = document_counter_dict[each_token]
            weight =  (1 + math.log10(tf)) *(getidf(each_token))
            document_weight_tf_dict[each_token] = weight
            maginitude = maginitude + (weight ** 2)

        nor_maginitude = math.sqrt(maginitude)
        document_cosine_dict = {}
        for each_token in document_weight_tf_dict.keys():
            tf = document_weight_tf_dict[each_token]
            document_cosine_dict[each_token] = tf / nor_maginitude

        final_score = 0
        for each_token in query_cosine_dict.keys():
            query_score = query_cosine_dict[each_token]
            document_score = 0
            if each_token in document_cosine_dict.keys():
                document_score = document_cosine_dict[each_token]
            final_score = final_score+query_score*document_score
        if final_score>=high_score:
            high_score = final_score
            document_ans = each_documet

    return document_ans,high_score

processed_collection = preprocess()
term_frequency_for_each_doc = get_term_frequency(processed_collection)

print("%.12f" % getidf('british'))
print("%.12f" % getidf('union'))
print("%.12f" % getidf('war'))
print("%.12f" % getidf('military'))
print("%.12f" % getidf('great'))

print("--------------")
print("%.12f" % getweight('02_washington_1793.txt','arrive'))
print("%.12f" % getweight('07_madison_1813.txt','war'))
print("%.12f" % getweight('12_jackson_1833.txt','union'))
print("%.12f" % getweight('09_monroe_1821.txt','british'))
print("%.12f" % getweight('05_jefferson_1805.txt','public'))
print("--------------")
print("(%s, %.12f)" % query("pleasing people"))
print("(%s, %.12f)" % query("british war"))
print("(%s, %.12f)" % query("false public"))
print("(%s, %.12f)" % query("people institutions"))
print("(%s, %.12f)" % query("violated willingly"))

