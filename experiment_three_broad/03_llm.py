# -*- coding: utf-8 -*-
"""
@date: 2024-10-24
@author: sap218
"""

import pandas as pd
from groq import Groq
import time
import numpy as np
import re

#####################

df = pd.read_csv("../spotify_user_reviews/exp3/DATASET_randomised_annotated_catch.csv")
list_of_posts = list(df["review"])


with open("../key/key.txt", "r") as k:
    groqkey = k.read()
del k

#####################

output_name = "../spotify_user_reviews/exp3/DATASET_randomised_annotated_catch_llm.csv"
stats_output_name = "../results/exp3/llm_stats.txt"

statistics = []

of_interest = "pay"

models_of_interest = [
    "gemma2-9b-it",
    "llama-3.1-8b-instant",
    "mixtral-8x7b-32768"
    ]

#####################

annotation_file = "../words_of_interest/%s.txt" % of_interest

words_of_interest = []
with open(annotation_file, "r") as t:
    for word in t:
        words_of_interest.append(word.strip("\n").strip(" "))
del t, word
words_of_interest = list(filter(None, words_of_interest))


''' loop over each model '''

run = ["broad","precise"]

for r in run:


    for model in models_of_interest:
        
        start_time = time.time()
        
        client = Groq(api_key=groqkey)
        
        ai_response = []
        for i in range(len(df)):
            print("Sentence iteration ", i+1, " out of ", len(df), " for ", of_interest, " & ", model)
        
            if r == "broad":
                request = 'The following sentence is a review of an App. Print a 1 or 0 if the sentence is related to any of these words "%s". Your response should only have the necessary data of a 1 or 0. Sentence: "%s"' % (", ".join(words_of_interest), list_of_posts[i])
            elif r == "precise":
                request = 'The following sentence is a review of an App. Print a 1 or 0 if the sentence is related to paying for the app. Your response should only have the necessary data of a 1 or 0. Sentence: "%s"' % list_of_posts[i]
            
            #print(request)
            
            ccc = client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": request
                        }
                ],
                model=model,
                seed=123,
            )
            response = ccc.choices[0].message.content
            #print(response)
            
            try: ai_response.append(int(response.strip()))
            except:
                #ai_response.append(np.nan) #print(response)
                
                ''' get first number available '''
                response = re.search(r'\d+', response).group()
                try: ai_response.append(int(response.strip()))
                except: ai_response.append(np.nan)
            
            del ccc
    
            
        COL_FOR_COUNT = "%s_%s" % (model.split('-')[0], r)
        df[COL_FOR_COUNT] = ai_response
        del COL_FOR_COUNT
        
        
        end_time = time.time() - start_time
        end_time = str(round(end_time, 2))
        statistics.append("%s %s %s time taken to annotate (seconds): %s" % (of_interest, model, r, end_time))
        del start_time, end_time
        
        
        del ai_response, i, model, client

#####################

''' outputs '''

df.to_csv(output_name, index=False, header=True)

with open(stats_output_name, 'w') as t:
    for word in statistics:
        t.write(word + '\n')
del t,word

#####################

del groqkey

# end of script
