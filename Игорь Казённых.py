#!/usr/bin/env python
# coding: utf-8

# # Прекод
# 
# # Сборный проект-4
# 
# Вам поручено разработать демонстрационную версию поиска изображений по запросу.
# 
# Для демонстрационной версии нужно обучить модель, которая получит векторное представление изображения, векторное представление текста, а на выходе выдаст число от 0 до 1 — покажет, насколько текст и картинка подходят друг другу.
# 
# ### Описание данных
# 
# Данные доступны по [ссылке](https://code.s3.yandex.net/datasets/dsplus_integrated_project_4.zip).
# 
# В файле `train_dataset.csv` находится информация, необходимая для обучения: имя файла изображения, идентификатор описания и текст описания. Для одной картинки может быть доступно до 5 описаний. Идентификатор описания имеет формат `<имя файла изображения>#<порядковый номер описания>`.
# 
# В папке `train_images` содержатся изображения для тренировки модели.
# 
# В файле `CrowdAnnotations.tsv` — данные по соответствию изображения и описания, полученные с помощью краудсорсинга. Номера колонок и соответствующий тип данных:
# 
# 1. Имя файла изображения.
# 2. Идентификатор описания.
# 3. Доля людей, подтвердивших, что описание соответствует изображению.
# 4. Количество человек, подтвердивших, что описание соответствует изображению.
# 5. Количество человек, подтвердивших, что описание не соответствует изображению.
# 
# В файле `ExpertAnnotations.tsv` содержатся данные по соответствию изображения и описания, полученные в результате опроса экспертов. Номера колонок и соответствующий тип данных:
# 
# 1. Имя файла изображения.
# 2. Идентификатор описания.
# 
# 3, 4, 5 — оценки трёх экспертов.
# 
# Эксперты ставят оценки по шкале от 1 до 4, где 1 — изображение и запрос совершенно не соответствуют друг другу, 2 — запрос содержит элементы описания изображения, но в целом запрос тексту не соответствует, 3 — запрос и текст соответствуют с точностью до некоторых деталей, 4 — запрос и текст соответствуют полностью.
# 
# В файле `test_queries.csv` находится информация, необходимая для тестирования: идентификатор запроса, текст запроса и релевантное изображение. Для одной картинки может быть доступно до 5 описаний. Идентификатор описания имеет формат `<имя файла изображения>#<порядковый номер описания>`.
# 
# В папке `test_images` содержатся изображения для тестирования модели.

# ## 1. Исследовательский анализ данных
# 
# Наш датасет содержит экспертные и краудсорсинговые оценки соответствия текста и изображения.
# 
# В файле с экспертными мнениями для каждой пары изображение-текст имеются оценки от трёх специалистов. Для решения задачи вы должны эти оценки агрегировать — превратить в одну. Существует несколько способов агрегации оценок, самый простой — голосование большинства: за какую оценку проголосовала большая часть экспертов (в нашем случае 2 или 3), та оценка и ставится как итоговая. Поскольку число экспертов меньше числа классов, может случиться, что каждый эксперт поставит разные оценки, например: 1, 4, 2. В таком случае данную пару изображение-текст можно исключить из датасета.
# 
# Вы можете воспользоваться другим методом агрегации оценок или придумать свой.
# 
# В файле с краудсорсинговыми оценками информация расположена в таком порядке:
# 
# 1. Доля исполнителей, подтвердивших, что текст **соответствует** картинке.
# 2. Количество исполнителей, подтвердивших, что текст **соответствует** картинке.
# 3. Количество исполнителей, подтвердивших, что текст **не соответствует** картинке.
# 
# После анализа экспертных и краудсорсинговых оценок выберите либо одну из них, либо объедините их в одну по какому-то критерию: например, оценка эксперта принимается с коэффициентом 0.6, а крауда — с коэффициентом 0.4.
# 
# Ваша модель должна возвращать на выходе вероятность соответствия изображения тексту, поэтому целевая переменная должна иметь значения от 0 до 1.
# 

# In[3]:


import torch.nn as nn
import torchvision.models as models
from torchsummary import summary

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Conv2D, BatchNormalization, MaxPooling2D, Dropout, Flatten
from tensorflow.keras.optimizers import Adam
from torchvision import transforms
import torch
import torch.nn as nn
import torchvision.models as models


from PIL import Image
import os

from tqdm import tqdm
from tqdm import notebook

import nltk
from nltk.corpus import stopwords as nltk_stopwords
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

import re

from sklearn.model_selection import GroupShuffleSplit

import os
import matplotlib.pyplot as plt


# In[4]:


from sklearn.pipeline import Pipeline

from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVR
from sklearn.preprocessing import  StandardScaler

from transformers import BertTokenizer, BertModel

import warnings

from sklearn.linear_model import LinearRegression
import catboost
from sklearn.metrics import mean_squared_error

import warnings


# In[5]:


warnings.filterwarnings('ignore')


# In[6]:


path = r'C:\Users\Igor\Downloads\dsplus_integrated_project_4\to_upload'


# In[7]:


CrowdAnnotations = pd.read_csv(path + '\\CrowdAnnotations.tsv',sep='\t', 
                               header = None, names=['image', 'query_id', 'result', 'Yes', 'No'])
CrowdAnnotations.head()


# In[8]:


CrowdAnnotations.info()


# In[9]:


ExpertAnnotations = pd.read_csv(path + '\\ExpertAnnotations.tsv',sep='\t', 
                               header = None, names=['image', 'query_id', 'Exp1', 'Exp2', 'Exp3'])
ExpertAnnotations.head()


# In[10]:


ExpertAnnotations.info()


# Количество оценок краудсорса гораздо выше количества оценок экспертов, при этом оценки экспертов должны быть более точны, т.к. имеют некоторую градацию, тогда как оценки краудсорса имеют бинарную природу.

# Эксперты ставят оценки по шкале от 1 до 4, где 1 — изображение и запрос совершенно не соответствуют друг другу, 2 — запрос содержит элементы описания изображения, но в целом запрос тексту не соответствует, 3 — запрос и текст соответствуют с точностью до некоторых деталей, 4 — запрос и текст соответствуют полностью.
# 
# Тогда переложим их градации в шкалу от 0 до 1, где
# 1 балл = 0
# 2 балла = 0,25
# 3 балла = 0,75
# 4 балла = 1
# 
# И определим средний балл в шкале от 0 до 1 от торех экспертов

# In[13]:


def condition(x):
    if x==1:
        return 0
    elif x==2:
        return 0.25
    elif x==3:
        return 0.75
    else:
        return 1

ExpertAnnotations['Exp1'] = ExpertAnnotations['Exp1'].apply(condition)
ExpertAnnotations['Exp2'] = ExpertAnnotations['Exp2'].apply(condition)
ExpertAnnotations['Exp3'] = ExpertAnnotations['Exp3'].apply(condition)


# In[14]:


ExpertAnnotations['result'] = (ExpertAnnotations['Exp1']+ExpertAnnotations['Exp2']+ExpertAnnotations['Exp3'])/3


# In[15]:


ExpertAnnotations.head()


# Соединяем экспертные и не экаспертный оценки и выводим итоговую. При этом за основу будем брать именно оценку экспертов ввиду их лучшего качества

# In[17]:


Annotations = pd.merge(ExpertAnnotations, CrowdAnnotations, on=['image', 'query_id'], how='left')


# In[18]:


Annotations.head()


# In[19]:


Annotations.info()


# In[20]:


Annotations.loc[Annotations['result_y'].isna(), ['result_y']] = Annotations['result_x']
Annotations.head()


# In[21]:


Annotations['itog'] = Annotations['result_x'] * 0.6 + Annotations['result_y'] * 0.4
Annotations.head()


# In[22]:


Annotations.info()


# In[23]:


Annotations = Annotations[['image','query_id', 'itog']]
Annotations.head()


# In[24]:


Annotations.describe()


# Соответствие фотографий описанию запроса как правило низкое

# ## 2. Проверка данных
# 
# В некоторых странах, где работает ваша компания, действуют ограничения по обработке изображений: поисковым сервисам и сервисам, предоставляющим возможность поиска, запрещено без разрешения родителей или законных представителей предоставлять любую информацию, в том числе, но не исключительно тексты, изображения, видео и аудио, содержащие описание, изображение или запись голоса детей. Ребёнком считается любой человек, не достигший 16 лет.
# 
# В вашем сервисе строго следуют законам стран, в которых работают. Поэтому при попытке посмотреть изображения, запрещённые законодательством, вместо картинок показывается дисклеймер:
# 
# > This image is unavailable in your country in compliance with local laws
# >
# 
# Однако у вас в PoC нет возможности воспользоваться данным функционалом. Поэтому все изображения, которые нарушают данный закон, нужно удалить из обучающей выборки.

# Перечень слов исключений

# In[28]:


exclude_words = [
'child',
'kid',
'baby',
'children',
'boy',
'girl',
'kids',
'babies',
'boys',
'girls'
]


# In[29]:


train_dataset = pd.read_csv(path +'\\train_dataset.csv')
train_dataset.info()


# In[30]:


train_dataset.head()


# In[31]:


train_dataset.duplicated().sum()


# In[32]:


train_dataset = train_dataset.loc[~train_dataset['query_text'].str.contains('|'.join(exclude_words))].reset_index(drop = True)


# In[33]:


train_dataset = pd.merge(train_dataset, Annotations, on=['image', 'query_id'], how='left')
train_dataset.head()


# In[34]:


train_dataset.info()


# In[35]:


len(train_dataset['image'].unique())


# При 4334 оценках всего 995 уникальных фотографий

# In[37]:


test_dataset = pd.read_csv(path + '\\test_queries.csv', sep='|')
test_dataset.info()


# In[38]:


test_dataset.head()


# In[39]:


len(test_dataset['image'].unique())


# При 500 оценках всего 100 уникальных фотографий

# Посмотрим в целом насколько релевантно описание содержимому фото в обучающей выборке

# In[ ]:





# In[42]:


df_bad = train_dataset.loc[train_dataset['itog']<0.5]
df_bad = df_bad.sample(5)
df_bad = df_bad.reset_index(drop = True)
df_bad


# In[43]:


for i in range(len(df_bad)):
    print(df_bad.loc[i,'query_text'])
    print(df_bad.loc[i,'itog'])
    plt.imshow(Image.open(path + '\\train_images\\' + df_bad.loc[i,'image']).convert('RGB'))
    plt.show()


# In[44]:


df_good = train_dataset.loc[train_dataset['itog']==1]
df_good = df_good.sample(5)
df_good = df_good.reset_index(drop = True)
df_good


# In[45]:


for i in range(len(df_good)):
    print(df_good.loc[i,'query_text'])
    print(df_good.loc[i,'itog'])
    plt.imshow(Image.open(path + '\\train_images\\' + df_good.loc[i,'image']).convert('RGB'))
    plt.show()


# В первом приближении все логично

# А еще ради интереса посмотрим, что в тестовой выборке

# In[48]:


df_test = test_dataset.sample(5)
df_test = df_test.reset_index(drop = True)
df_test


# In[49]:


for i in range(len(df_test)):
    print(df_test.loc[i,'query_text'])
    plt.imshow(Image.open(path + '\\test_images\\' + df_test.loc[i,'image']).convert('RGB'))
    plt.show()


# Изображение и описание полностью совпадают

# ## 3. Векторизация изображений
# 
# Перейдём к векторизации изображений.
# 
# Самый примитивный способ — прочесть изображение и превратить полученную матрицу в вектор. Такой способ нам не подходит: длина векторов может быть сильно разной, так как размеры изображений разные. Поэтому стоит обратиться к свёрточным сетям: они позволяют "выделить" главные компоненты изображений. Как это сделать? Нужно выбрать какую-либо архитектуру, например ResNet-18, посмотреть на слои и исключить полносвязные слои, которые отвечают за конечное предсказание. При этом можно загрузить модель данной архитектуры, предварительно натренированную на датасете ImageNet.

# In[52]:


resnet = models.resnet18(pretrained=True)

for param in resnet.parameters():
    param.requires_grad_(False) 
    
modules = list(resnet.children())[:-1]
resnet = nn.Sequential(*modules) 

resnet.eval()  


# In[53]:


norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    norm,
]) 


# In[54]:


def vectorize(data_txt_itog, data_img, data_path):
    output_tensor_train = []
    
    for i in tqdm(data_img):   
        img = Image.open(data_path +i).convert('RGB') 
        image_tensor = preprocess(img)
        if len(output_tensor_train) == 0:
            output_tensor_train = pd.DataFrame(resnet(image_tensor.unsqueeze(0)).flatten()).transpose()
        else:
            output_tensor_train = pd.concat([output_tensor_train, pd.DataFrame(resnet(image_tensor.unsqueeze(0)).flatten()).transpose()])

    output_tensor_train = output_tensor_train.reset_index(drop = True)
    cols = []
    for i in output_tensor_train.columns:
        cols.append('img_'+str(i))
    output_tensor_train.columns = cols

    output_tensor_train = pd.concat([data_txt_itog,output_tensor_train], axis = 1)
    
    return output_tensor_train


# In[55]:


output_tensor_train = vectorize(train_dataset[['image','query_text','itog']], train_dataset['image'], path + '\\train_images\\')


# In[56]:


output_tensor_train.head()


# ## 4. Векторизация текстов
# 
# Следующий этап — векторизация текстов. Вы можете поэкспериментировать с несколькими способами векторизации текстов:
# 
# - tf-idf
# - word2vec
# - \*трансформеры (например Bert)
# 
# \* — если вы изучали трансформеры в спринте Машинное обучение для текстов.
# 

# #### Стандартный подход

# In[59]:


m = WordNetLemmatizer() 


# In[60]:


nltk.download('stopwords')
stopwords = list(set(nltk_stopwords.words('english')))


# In[61]:


def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)


# In[62]:


def text_tokenize(text):
    #tokenized = nltk.word_tokenize(text)
    tokenized =  [m.lemmatize(w, get_wordnet_pos(w)) for w in nltk.word_tokenize(text)]
    joined = ' '.join(tokenized)
    text = re.sub(r'[^a-zA-Z]', ' ', joined)
    text = ' '.join(text.split())
    #text =  [m.lemmatize(w, get_wordnet_pos(w)) for w in nltk.word_tokenize(text)]
    return text


# In[63]:


train_dataset=train_dataset.reset_index(drop=True)


# In[64]:


txt = text_tokenize(train_dataset['query_text'][0])
txt


# In[65]:


notebook.tqdm.pandas() 
train_dataset['text_final'] = train_dataset['query_text'].progress_apply(text_tokenize)


# In[66]:


train_dataset.head()


# In[67]:


#Подгрузка векторизация
count_tf_idf = TfidfVectorizer(stop_words=stopwords)
tf_idf = count_tf_idf.fit_transform(train_dataset['text_final'])


# In[68]:


x= tf_idf.toarray()
x[0]


# Какие-то нули один получились. Попробуем Bert

# #### Bert

# In[71]:


bert_model = BertModel.from_pretrained('bert-base-uncased',
                                       output_hidden_states=True)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


# In[72]:


def transform_text(column):
    tokenized = column.apply(lambda x: tokenizer.encode(x, add_special_tokens=True))

    max_len = 0
    for i in tokenized.values:
        if len(i) > max_len:
            max_len = len(i)

    padded = np.array([i + [0]*(max_len - len(i)) for i in tokenized.values])
    attention_mask = np.where(padded != 0, 1, 0)
    
    batch_size = 100
    embeddings = []
    batch_count = padded.shape[0] // batch_size + 1
    for i in tqdm(range(batch_count), disable=batch_count==1):
        batch = torch.LongTensor(padded[batch_size*i:batch_size*(i+1)]) 
        attention_mask_batch = torch.LongTensor(attention_mask[batch_size*i:batch_size*(i+1)])
        with torch.no_grad():
            batch_embeddings = bert_model(batch, attention_mask=attention_mask_batch)
        
        embeddings.append(batch_embeddings[0][:,0,:].numpy())
    df = np.concatenate(embeddings)
    df = pd.DataFrame(df)

    cols = []
    for i in df.columns:
        cols.append('txt_'+str(i))
    df.columns = cols
    
    return df


# In[73]:


train_dataset['query_text']


# In[74]:


text_features = transform_text(train_dataset['query_text'])
text_features.shape


# In[75]:


text_features.head(10)


# In[76]:


text_features = pd.concat([train_dataset[['image','query_text','itog']],text_features], axis = 1)
text_features.head(10)


# Вроде поинтереснее

# ## 5. Объединение векторов

# In[79]:


text_features.iloc[:,3:]


# In[80]:


uni = pd.concat([output_tensor_train,text_features.iloc[:,3:]], axis = 1)
uni.head()


# ## 6. Обучение модели предсказания соответствия

# In[82]:


RANDOM_STATE=42


# In[83]:


X = uni.drop(['query_text','itog', 'image'], axis = 1)
y = uni['itog']


# In[84]:


gss = GroupShuffleSplit(n_splits=1, train_size=.7, random_state=RANDOM_STATE)


# In[85]:


train_indices, test_indices = next(gss.split(X, y, groups=uni['image']))


# In[86]:


X_train = X.iloc[train_indices, :]
X_test = X.iloc[test_indices, :]
y_train = y.iloc[train_indices]
y_test = y.iloc[test_indices]


# In[87]:


X_train.shape


# In[88]:


X_train.head()


# In[89]:


y_train.shape


# In[90]:


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# ### Строим модель

# #### Линейная регрессия

# In[93]:


lr_model = LinearRegression()
lr_model.fit(X_train, y_train)


# In[94]:


prediction = lr_model.predict(X_test)


# In[95]:


rmse = mean_squared_error(y_test, prediction) ** 0.5
rmse


# Многовато

# #### CatBoost

# In[98]:


train_pool = catboost.Pool(X_train, y_train)
val_pool = catboost.Pool(X_test, y_test)
cb_model = catboost.CatBoostRegressor(eval_metric='RMSE', verbose=100, random_seed=RANDOM_STATE)
cb_model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=150)


# Явно лучше.

# ## 7. Тестирование модели

# In[100]:


#test_dataset = test_dataset['image'].drop_duplicates()
len(test_dataset)


# In[101]:


test_dataset = test_dataset.reset_index(drop = True)
test_dataset = test_dataset.drop(['Unnamed: 0'], axis = 1)
test_dataset.head()


# In[102]:


preview_df = test_dataset.sample(20, random_state=RANDOM_STATE)
preview_df = preview_df.reset_index(drop = True)
preview_df


# Суть:
# - векторизируем текст 1-ого запроса (TXT_1)
# - векторизируем каждое тестовое изображение (IMG_i)
# - соединияем два вектора (TXT_1 + IMG_1; TXT_1 + IMG_2...)
# - получаем предсказательную метрику для каждого соединения
# - вычисляем максимальную предсказательную метрику, исходя из чего определяем IMG_?  

# #### Векторизируем тестовые изображения

# In[105]:


output_tensor_test = vectorize(test_dataset[['image','query_text']], test_dataset['image'], path + '\\test_images\\')


# In[106]:


output_tensor_test.head()


# In[107]:


img = output_tensor_test.drop(['query_text'],axis = 1)


# In[108]:


img = img.drop_duplicates()


# In[109]:


img = img.reset_index(drop = True)
img


# In[125]:


clean_img = img[['image']]


# #### Векторизируем отобранные запросы

# In[111]:


txt = transform_text(preview_df['query_text'])


# In[112]:


txt


# In[113]:


txt_full = pd.concat([preview_df[['query_text']],txt], axis = 1)
txt_full.head(10)


# In[119]:


img.iloc[0:1,1:]


# #### Тестируем

# In[135]:


def checking(txt, img):
    txt['query_text'] = txt['query_text'].str.lower()
    if len(txt.loc[txt['query_text'].str.contains('|'.join(exclude_words))]):
        print('This image is unavailable in your country in compliance with local laws.')
        return
    else:
        print(txt['query_text'])
        txt = txt.drop('query_text',axis=1)
        

    df_full = []
    for i in range(len(img)):
        df = pd.concat([img.iloc[i:i+1,1:].reset_index(drop=True),txt.reset_index(drop=True)], axis=1)
        if len(df_full) == 0:
            df_full = df
        else:
            df_full = pd.concat([df_full,df])
    df_full = df_full.reset_index(drop=True)
    test_features = scaler.transform(df_full)
    predictions = cb_model.predict(test_features)
    best_idx = np.argmax(predictions)
    plt.imshow(Image.open(path + '\\test_images\\' + clean_img.loc[best_idx,'image']).convert('RGB'))
    plt.show()

    #return df_full


# In[137]:


for i in range(len(txt_full)):
    x = txt_full.iloc[i:i+1, :]
    checking(x,img)


# В первом приближении моделька худо-бедно выдает подходящие результаты
