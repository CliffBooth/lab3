import re

import pandas as pd

captions = pd.read_csv('flickr8k/captions.txt')
captions['image'] = captions['image'].apply(lambda x: f'flickr8k/images/{x}')

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub('\s+', ' ', text)
    text = text.strip()
    text = '[start] ' + text + ' [end]'
    return text

captions['caption'] = captions['caption'].apply(preprocess)