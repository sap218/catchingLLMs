# -*- coding: utf-8 -*-
"""
@date: 2024-10-24
@author: sap218
"""

corpus = "../spotify_user_reviews/exp3/DATASET_randomised_annotated.csv"
filter_level = "light" # "heavy"

import pandas as pd
import re
import time
import json

import spacy
from spacy.matcher import PhraseMatcher
nlp = spacy.load("en_core_web_sm")

#####################

from highlevel import *

''' stopWords '''
if filter_level == "light": stopWords = stopWords[0]
elif filter_level == "heavy": stopWords = stopWords[1]

stopWords_lemma = []
for word in stopWords:
    word = cleantext(word.lower())
    doc = nlp(word)
    doc_lemma = " ".join([token.lemma_ for token in doc])
    stopWords_lemma.append(doc_lemma)
stopWords_lemma_filt = list(filter(None, stopWords_lemma))
stopWords_lemma_filt_flat = [word for phrase in stopWords_lemma_filt for word in phrase.split()]

stopWords = list(set(stopWords_lemma_filt_flat))
del word, doc, doc_lemma, stopWords_lemma, stopWords_lemma_filt, stopWords_lemma_filt_flat

#####################

df = pd.read_csv(corpus)
df = df.fillna("0")
list_of_posts = list(df["review"])

post_stats = []
for post in list_of_posts:
    post = post.split()
    post_stats.append(len(post))
del post

output_name = "../spotify_user_reviews/exp3/DATASET_randomised_annotated_catch.csv"
stats_output_name = "../results/exp3/catch_stats.txt"

statistics = []

#categories = ["pay"]

#####################

#for of_interest in categories: 

of_interest = "pay"    


annotation_file = "../words_of_interest/%s.txt" % of_interest


words_of_interest = []
with open(annotation_file, "r") as t:
    for word in t:
        words_of_interest.append(word.strip("\n").strip(" "))
del t, word
words_of_interest = list(filter(None, words_of_interest))


statistics.append("%s concepts count: %s" % (of_interest, len(words_of_interest)))
statistics.append("%s post count: %s" % (of_interest, len(list_of_posts)))
statistics.append("%s average word count: %s" % (of_interest, sum(post_stats)/len(post_stats) ))



''' cleaning words of interest '''
##words_of_interest_clean_lemma_stpwrd = [] 
concept_patterns = [] # for matcher

# preprocess concepts: Lemmatize & stopWords
for concept in words_of_interest: 
    concept = cleantext(concept.lower())
    
    doc = nlp(concept)
    
    ## lemma
    doc_lemma = [token.lemma_ for token in doc]
    ## stopwords
    doc_lemma_stpwrd = [remove_stop_words(text, stopWords) for text in doc_lemma]
    doc_lemma_stpwrd = list(filter(None, doc_lemma_stpwrd))
    
    if doc_lemma_stpwrd:
        concept_patterns.append(nlp(" ".join(doc_lemma_stpwrd).lower()))
        ##words_of_interest_clean_lemma_stpwrd.append(" ".join(doc_lemma_stpwrd).lower())
    
del concept, doc, doc_lemma, doc_lemma_stpwrd

matcher = PhraseMatcher(nlp.vocab) # initialize phrase matcher
matcher.add("Concepts", None, *concept_patterns) # convert concepts into patterns
del concept_patterns


''' annotation '''

start_time = time.time()

#matched_output_list = []
#list_of_posts_clean_lemma_stpwrd = []

FOR_THE_COUNT = []
    
x = 0
y = 0
for post in list_of_posts:
    x = x + 1
    print("Sentence iteration ", x, " out of ", len(list_of_posts))
    post = cleantext(post.lower())
    
    doc = nlp(post)
    
    ## lemma
    doc_lemma = [token.lemma_ for token in doc]
    ## stopwords
    doc_lemma_stpwrd = [remove_stop_words(text, stopWords) for text in doc_lemma]
    doc_lemma_stpwrd = list(filter(None, doc_lemma_stpwrd))
        
    #list_of_posts_clean_lemma_stpwrd.append(" ".join(doc_lemma_stpwrd).lower())
    
    doc = nlp(" ".join(doc_lemma_stpwrd).lower())
    matches = matcher(doc)
    
    if matches:
        '''
        matched_concepts = set()
        for match_id, start, end in matches:
            matched_span = doc[start:end]
            matched_concepts.add(matched_span.text)
            
        matched_output_list.append([ list(matched_concepts), list_of_posts[y] ])
        
        del matched_concepts, match_id, start, end, matched_span
        '''
        
        FOR_THE_COUNT.append(1)
        
    else: 
        '''
        matched_output_list.append([ "NO ANNOTATION", list_of_posts[y] ])
        '''
        
        FOR_THE_COUNT.append(0)
    
    y = y + 1

end_time = time.time() - start_time
end_time = str(round(end_time, 2))
print("Seconds taken to annotate %s: %s" % (of_interest, end_time))

statistics.append("%s broad time taken to annotate (seconds): %s" % (of_interest, end_time))

del x, y, post, doc, doc_lemma, doc_lemma_stpwrd, matches
del start_time, end_time


COL_FOR_COUNT = "catch_broad"
df[COL_FOR_COUNT] = FOR_THE_COUNT

del FOR_THE_COUNT, COL_FOR_COUNT

#####################
#####################

words_of_interest = ["pay"]

''' cleaning words of interest '''
concept_patterns = [] # for matcher

# preprocess concepts: Lemmatize & stopWords
for concept in words_of_interest: 
    concept = cleantext(concept.lower())
    
    doc = nlp(concept)
    
    ## lemma
    doc_lemma = [token.lemma_ for token in doc]
    ## stopwords
    doc_lemma_stpwrd = [remove_stop_words(text, stopWords) for text in doc_lemma]
    doc_lemma_stpwrd = list(filter(None, doc_lemma_stpwrd))
    
    if doc_lemma_stpwrd:
        concept_patterns.append(nlp(" ".join(doc_lemma_stpwrd).lower()))
del concept, doc, doc_lemma, doc_lemma_stpwrd

matcher = PhraseMatcher(nlp.vocab) # initialize phrase matcher
matcher.add("Concepts", None, *concept_patterns) # convert concepts into patterns
del concept_patterns


''' annotation '''

start_time = time.time()

FOR_THE_COUNT = []
    
x = 0
y = 0
for post in list_of_posts:
    x = x + 1
    print("Sentence iteration ", x, " out of ", len(list_of_posts))
    post = cleantext(post.lower())
    
    doc = nlp(post)
    
    ## lemma
    doc_lemma = [token.lemma_ for token in doc]
    ## stopwords
    doc_lemma_stpwrd = [remove_stop_words(text, stopWords) for text in doc_lemma]
    doc_lemma_stpwrd = list(filter(None, doc_lemma_stpwrd))
        
    doc = nlp(" ".join(doc_lemma_stpwrd).lower())
    matches = matcher(doc)
    
    if matches:        
        FOR_THE_COUNT.append(1)
    else: 
        FOR_THE_COUNT.append(0)
    
    y = y + 1

end_time = time.time() - start_time
end_time = str(round(end_time, 2))
print("Seconds taken to annotate %s: %s" % (of_interest, end_time))

statistics.append("%s precise time taken to annotate (seconds): %s" % (of_interest, end_time))

del x, y, post, doc, doc_lemma, doc_lemma_stpwrd, matches
del start_time, end_time

COL_FOR_COUNT = "catch_precise"
df[COL_FOR_COUNT] = FOR_THE_COUNT

del FOR_THE_COUNT, COL_FOR_COUNT

#####################
#####################

''' outputs '''

with open(stats_output_name, 'w') as t:
    for word in statistics:
        t.write(word + '\n')
del t,word


df.to_csv(output_name, index=False)

#####################

# end of script
