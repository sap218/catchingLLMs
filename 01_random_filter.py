# -*- coding: utf-8 -*-
"""
@date: 2024-10-11
@author: sap218
"""

import pandas as pd

df = pd.read_csv("spotify_user_reviews/DATASET.csv")

df_filter = df.sample(n=100, random_state=123)

df_filter.to_csv("spotify_user_reviews/DATASET_randomised.csv",index=False)

#####################

# end of script
