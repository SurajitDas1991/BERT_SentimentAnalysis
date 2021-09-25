from typing import no_type_check


import pandas as pd
import numpy as np
import ktrain
import tensorflow as tf
from ktrain import text

df_train=pd.read_excel('train.xlsx',dtype=str)
print(df_train.head())

df_test=pd.read_excel('test.xlsx',dtype=str)

print(df_test.head())

train,val,preprocess= text.texts_from_df(train_df=df_train,text_column='Reviews',label_columns='Sentiment',val_df=df_test,maxlen=400,preprocess_mode='distilbert')

model=text.text_classifier(name='distilbert',train_data=train,preproc=preprocess)

learner = ktrain.get_learner(model,
                             train_data=train,
                             val_data=val,
                             batch_size=6)

# find good learning rate
#learner.lr_find()             # briefly simulate training to find good learning rate
#learner.lr_plot()             # visually identify best learning rate

# train using 1cycle learning rate schedule for 3 epochs
learner.fit_onecycle(2e-5, 1)

predictor = ktrain.get_predictor(learner.model, preprocess)

data = [ 'This movie was horrible! The plot was boring. Acting was okay, though.',
         'The film really sucked. I want my money back.',
        'The plot had too many holes.',
        'What a beautiful romantic comedy. 10/10 would see again!',
         ]

print(predictor.predict(data))

predictor.save('model')

