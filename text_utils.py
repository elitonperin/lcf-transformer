import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import numpy as np
import pandas as pd
from polyglot.detect import Detector
from langdetect import detect
from polyglot.detect.base import logger as polyglot_logger
from tqdm import tqdm
tqdm.pandas()
polyglot_logger.setLevel("ERROR")

null_aspects = ['!', '#', '@', "'", '~', '^', '\n', '`', '%', '"', '“', '´', '’', ',', '.', ':', ';', '?', '/']
symbols = ['!', '#', '@','~', '^', '`', '\n', '%', '"', "\\"]

stop_words = set(stopwords.words('english'))
stop_words.remove('up')
stop_words.remove('down')
stop_words.remove('in')
stop_words.remove('out')

strip_re_components = []
for sw in stop_words:
    escaped_sw = re.escape(sw)
    strip_re_components.append(fr'(?:^{escaped_sw}\b)')
    strip_re_components.append(fr'(?:\b{escaped_sw}$)')
strip_re_components.append(r'(?:\b[,.:;/?$%#@!]+\b)')
strip_re = re.compile('|'.join(strip_re_components), re.I)

def remove_stop_words(aspect_list):
    clean = (strip_re.sub('', a).strip() for a in aspect_list)
    filtered = filter(None, clean)
    return list(filtered)

def clean_aspects(aspects):
  return [remove_stop_words(a) for a in aspects]

def only_lang_(text, limiar=0.9, lang='en'):
  try:
    res = detect_langs(text)
  except:
    #print(text)
    return None
  print(res)
  if res[0].lang == lang and res[0].prob >= limiar:
    return text
  return None

def only_lang(text, limiar=0.9, lang='en'):
  try:
    res = Detector(text, quiet=True)
  except:
    #print(text)
    return None
  if res.language.code == lang and res.language.confidence >= limiar:
    return text
  return None

def transform_sentence_in_T(aspect, text):
  if aspect in null_aspects:
    return None
  if aspect is None or text is None:
    return None
  if aspect in text:
    return text.replace(str(aspect), '$T$')
  else:  
    try:
      t_sent = re.sub(str(aspect), '$T$', text, flags=re.IGNORECASE)
      return t_sent
    except:
      print(aspect, '---', text)
      return None

def transform_data_in_T(data):
  t_sents = []
  for a, s in zip(data['aspect_name'], data['snippet']):  
    t_sent = transform_sentence_in_T(a, s)
    t_sents.append(t_sent)
  data['T_sent'] = t_sents
  return data

def preprocessing_snippet(x, lang='en'):
  x = str(x)
  x = x.replace('[This review was collected as part of a promotion.]', '')
  for s in symbols:
    x = x.replace(s, '')
  x.replace('\n', '')
  if lang:
    x = only_lang(x, lang='en')
  if x == '':
    x = None
  return x

def preprocessing_aspects(x, lang='en'):
  x = str(x)
  strip_re.sub('', x)
  x = re.sub(r'\B\W\B', '', x)
  x = re.sub(r"\B's", '', x)
  x = re.sub(r"\B’s", '', x)
  for s in null_aspects:
    x = x.replace(s, ' ')
  x = x.strip()
  x.replace('\n', '')
  # if lang:
  #   x = only_lang(x, lang='en')
  if x == '':
    x = None
  return x

def preprocessing_dataframes(data, lang='en'):
  data.drop_duplicates(subset=['snippet', 'aspect_name'], inplace=True)
  data['snippet'] = data['snippet'].progress_apply(preprocessing_snippet, lang=lang)
  data['aspect_name'] = data['aspect_name'].progress_apply(preprocessing_aspects, lang=lang)
  data.drop_duplicates(subset=['snippet', 'aspect_name'], inplace=True)
  return data

def save_xml_seg(data, file_name='train', frac=1):
  new_data = data[[ 'T_sent', 'aspect_name', 'sentiment']]
  # new_data['T_sent'] = new_data['T_sent'].str.replace('\n', '')
  new_data = new_data.replace('\\n', ' ', regex=True)
  new_data = new_data.dropna()
  new_data['sentiment'] = new_data['sentiment'].apply(int)
  new_data = new_data.sample(frac=frac).reset_index(drop=True)
  np.savetxt(file_name + '.txt', new_data.values, fmt='%s\n%s\n%s')
  return new_data

def save_files(data, file_name, frac=1):
  print('Dropping nan values...')
  new_data = data.dropna()
  print('Reseting index values...')
  new_data = new_data.reset_index(drop=True)
  print('Saving csv')
  new_data.to_csv(file_name + '.csv.zip', compression='zip')
  print('Creating seg.xml file')
  save_xml_seg(new_data, file_name=file_name, frac=frac)
  return new_data

def select_max_n_elem_dframe_by_column(data, column, n):
    return data.groupby(column, group_keys=False).apply(lambda x: x.sample(min(len(x), n)))

def select_min_n_elem_dframe_by_column(data, column, n):
    return data.groupby(column).filter(lambda x: len(x) >= n)
