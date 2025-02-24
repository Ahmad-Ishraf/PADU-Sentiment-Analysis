# %%
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn import preprocessing
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split

from transformers import BertTokenizerFast
from transformers import BertTokenizer
from transformers import TFBertModel
from tokenizers import Tokenizer

import tensorflow as tf
from tensorflow import keras

from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import classification_report, multilabel_confusion_matrix, confusion_matrix, ConfusionMatrixDisplay

import matplotlib.pyplot as plt
import seaborn as sns

seed = 42

# %%
physical_devices = tf.config.list_physical_devices('GPU')

# %%
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

# %%
df = pd.read_csv('C:/Users/User/Desktop/augmented_with_original_labeled_data.csv')
print(df.to_markdown())

# %%
X = df['clean_text']
y = df['sentiment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

# %%
data_train = {'clean_text':X_train, 'sentiment':y_train}

df_train = pd.DataFrame(data_train)

# %%
df_train['sentiment'].value_counts()

# %%
#Fitting
ros = RandomOverSampler()
train_x, train_y = ros.fit_resample(np.array(df_train['clean_text']).reshape(-1, 1), np.array(df_train['sentiment']).reshape(-1, 1));
train_os = pd.DataFrame(list(zip([x[0] for x in train_x], train_y)), columns = ['clean_text', 'sentiment']);

# %%
train_os['sentiment'].value_counts()

# %%
X = train_os['clean_text'].values
y = train_os['sentiment'].values

# %%
#Data Splitting
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)

# %%
y_train_le = y_train.copy()
y_valid_le = y_valid.copy()
y_test_le = y_test.copy()

# %%
#Encoding
ohe = preprocessing.OneHotEncoder()
y_train = ohe.fit_transform(np.array(y_train).reshape(-1, 1)).toarray()
y_valid = ohe.fit_transform(np.array(y_valid).reshape(-1, 1)).toarray()
y_test = ohe.fit_transform(np.array(y_test).reshape(-1, 1)).toarray()

# %%
print(f"TRAINING DATA: {X_train.shape[0]}\nVALIDATION DATA: {X_valid.shape[0]}\nTESTING DATA: {X_test.shape[0]}" )

# %%
MAX_LEN=128

# %%
#Load tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')

# %%
#Tokenization
def tokenize(data,max_len) :
    input_ids = []
    attention_masks = []
    for i in range(len(data)):
        encoded = tokenizer.encode_plus(
            data[i],
            add_special_tokens=True,
            max_length=max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True
        )
        input_ids.append(encoded['input_ids'])
        attention_masks.append(encoded['attention_mask'])
    return np.array(input_ids),np.array(attention_masks)

# %%
#Reset index
X_test = X_test.reset_index(drop=True)

# %%
#Tokenize the data
train_input_ids, train_attention_masks = tokenize(X_train, MAX_LEN)
val_input_ids, val_attention_masks = tokenize(X_valid, MAX_LEN)
test_input_ids, test_attention_masks = tokenize(X_test, MAX_LEN)

# %%
#Load model
bert_model = TFBertModel.from_pretrained('bert-base-multilingual-uncased')

# %%
#Define the model
def create_model(bert_model, max_len=MAX_LEN):
    
    ##params###
    opt = tf.keras.optimizers.legacy.Adam(learning_rate=1e-5, decay=1e-7)
    loss = tf.keras.losses.CategoricalCrossentropy()
    accuracy = tf.keras.metrics.CategoricalAccuracy()


    input_ids = tf.keras.Input(shape=(max_len,),dtype='int32')
    
    attention_masks = tf.keras.Input(shape=(max_len,),dtype='int32')
    
    embeddings = bert_model([input_ids,attention_masks])[1]
    
    output = tf.keras.layers.Dense(3, activation="softmax")(embeddings)
    
    model = tf.keras.models.Model(inputs = [input_ids,attention_masks], outputs = output)
    
    model.compile(opt, loss=loss, metrics=accuracy)
    
    
    return model

# %%
#Create the model
model = create_model(bert_model, MAX_LEN)
model.summary()

# %%
print(train_input_ids.shape)
print(train_attention_masks.shape)
print(y_train.shape)

# %%
print(val_input_ids.shape)
print(val_attention_masks.shape)
print(y_valid.shape)

# %%
print(np.max(train_input_ids))
print(np.max(val_input_ids))

# %%
#Train the model
try:
    history_bert = model.fit([train_input_ids,train_attention_masks], y_train, validation_data=([val_input_ids,val_attention_masks], y_valid), epochs=8, batch_size=8)
except Exception as e:
    print("Error:",e)

# %%
result_bert = model.predict([test_input_ids,test_attention_masks])

# %%
y_pred_bert =  np.zeros_like(result_bert)
y_pred_bert[np.arange(len(y_pred_bert)), result_bert.argmax(1)] = 1

# %%
print('\tClassification Report:\n\n',classification_report(y_test,y_pred_bert, target_names=['Negative', 'Neutral', 'Positive']),'\nAccuracy Score:',accuracy_score(y_test,y_pred_bert))

# %%
mlcm = multilabel_confusion_matrix(y_test,y_pred_bert)

# %%
print(mlcm)

# %%
labels = ['Negative', 'Neutral', 'Positive']
f, axes = plt.subplots(1, 3, figsize=(15, 5))
axes = axes.ravel()
for i in range(y_test.shape[1]):
    disp = ConfusionMatrixDisplay(confusion_matrix(y_test[:, i],
                                                   y_pred_bert[:, i]))
    disp.plot(ax=axes[i], values_format='.4g')
    disp.ax_.set_title(f'{labels[i]}')
    if i<10:
        disp.ax_.set_xlabel('')
    if i%5!=0:
        disp.ax_.set_ylabel('')
    disp.im_.colorbar.remove()

plt.subplots_adjust(wspace=0.10, hspace=0.1)
f.colorbar(disp.im_, ax=axes)
plt.show()