import pandas as pd
import numpy as np
import re
# !pip install transformers

"der kommi sollte nur im test_branch, nicht aber im master auftauchen"

file_path = '/Users/philippwarter/Desktop/MAPR - Forschungsmaster/Projektarbeit 2. Semester/Tag Mapping Daten/mapped_7k_clean.xlsx'
sheet_name = '8k_train'
df = pd.read_excel(file_path, sheet_name=sheet_name)



from transformers.models.roberta_prelayernorm.modeling_roberta_prelayernorm import RobertaPreLayerNormEncoder
if 'gt öfter vorhanden?' in df.columns:
    df = df.drop('gt öfter vorhanden?', axis=1)


# strings
df['gt'] = df['gt'].astype(str)
df['lt'] = df['lt'].astype(str)

df['desc_gt'] = df['desc_gt'].astype(str)
df['desc_lt'] = df['desc_lt'].astype(str)

df['u_gt'] = df['u_gt'].astype(str)
df['u_lt'] = df['u_lt'].astype(str)


# pattern (side_id und alles nach komma weg)
pattern_sid = r'[A-Z]{2}\d{6}'

pattern_descgt = r',[^,]*$'

pattern_machine = r':[A-Za-z]{3,7}\d{1,2}'

pattern_ = r'_[A-Z]{0,3}'


#%%

machines_from_master = [
    'BAC',
    'CAB',
    'CBoost',
    'CGen',
    'CO2C',
    'COC',
    'CTurb',
    'CWP',
    'FeedC',
    'FGB',
    'GANC',
    'Gen',
    'GOXC',
    'H2C',
    'LPBoost',
    'LPGen',
    'LPTurb',
    'MAC',
    'NH3C',
    'PAIRC',
    'RecC',
    'Ref',
    'RefC',
    'SynC',
    'VP',
    'WBoost',
    'WGen',
    'WTurb',
    'LiqTurb',
    'SteamTurb',
    'GasTurb',
    'CTF',
    'LOXP',
    'LINP',
    'LARP',
    'EJF',
    'CCM',
    'IPC',
    'CWS',
    'DCAC',
    'DCWC',
    'FeedPT',
    'GSFR',
    'MHE',
    'PPU',
    'PreRef',
    'PSA',
    'Reform',
    'RHE',
    'Shift',
    'SWGR',
    'Tank',
    'Tank',
    'TRF',
    'Stripper',
    'Liquifier',
    'PreCl',
    'Chlr',
    'Cond',
    'Rcv',
    'Econ',
    'SubCl',
    'Dryer',
    'Catox',
    'LPCol',
    'MPCol',
    'HPCol',
    'CArCol',
    'PArCol',
    'PAirC'
    'KrXeCol',
    'SpargeTank',
    'LOXTank',
    'LOXVAP',
    'LINTANK',
    'LARTank',
    'LINVAP',
    'RECC',
    'LOXTANK',
    'GOX',
    'ASC',
    'GAN',
]


##### Versuch, alle Machines mit dem muster zu erreichen. Problem: mit einer Zahl dahinter erkennt er sie, mit zwei nicht####

machine_tag = []
for index, entry in enumerate(df['gt']):
    match = None
    for machine in machines_from_master:
        pattern = rf'\b{machine}\d'
        if re.search(pattern, entry):
            match = re.findall(pattern, entry)
            break
    if match:
        machine_tag.append(match[0])
        df.at[index, 'gt'] = df.at[index, 'gt'].replace(match[0], '')


#%%
# Create a DataFrame from the machine_tag list
df_machine_tag = pd.DataFrame(machine_tag, columns=['Machine Tag'])

# Replace the ':' character in the 'Machine Tag' column # der hier löscht wieder nur nach dem standard pattern ohne machine_master
df_machine_tag['Machine Tag'] = df_machine_tag['Machine Tag'].str.replace(':', '')

df_machine_tag['Machine Tag'] = df_machine_tag['Machine Tag'].dropna()
df_machine_tag.dropna(subset=['Machine Tag'], inplace=True)


# aus 6 spalten 2 machen

df['gt'] = df.apply(lambda row: ' '.join([str(row['gt'])]), axis=1)

df['lt'] = df.apply(lambda row: ' '.join([str(row['lt']), str(row['desc_lt']), str(row['u_lt'])]), axis=1)


# restliche spalten löschen

df = df.drop(['desc_gt', 'u_gt', 'desc_lt', 'u_lt'], axis=1)

#%%
df['gt'] = df['gt'].str.replace(pattern_sid, '')

df['gt'] = df['gt'].str.replace(':', '')

# notgedrungen scheiß funktion die die erste digit löscht:

            # unbedingt fixen!!

def remove_first_digit(entry):
    if entry[0].isdigit():
        return entry[1:]
    return entry

df['gt'] = df['gt'].apply(remove_first_digit)

unique_entries = df['gt'].unique()

import tensorflow as tf
from transformers import TFBertModel, BertTokenizer
from transformers import RobertaTokenizer, TFRobertaModel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# Tokenize and encode your text data using BERT tokenizer
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
# tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

max_length = 50  # Maximum sequence length for BERT
X_encoded = []

for text in df['lt']:
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='np'
    )
    X_encoded.append(encoding)

import numpy as np


# Tokenize and encode your text data using BERT tokenizer
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
max_length = 50  # Set your desired max sequence length here

X_encoded = []
for text in df['lt']:
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='np'
    )
    X_encoded.append(encoding)

# Convert the list of encoded inputs to numpy arrays
X_input = np.concatenate([encoding['input_ids'] for encoding in X_encoded])
X_attention = np.concatenate([encoding['attention_mask'] for encoding in X_encoded])

# label encode gt
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['gt'])

# Split the data into training and testing sets (which you already did manually)
X_train, X_test, y_train, y_test = train_test_split(X_input, y, test_size=0.1, random_state=42)

# Split the attention masks into training and testing sets (which should match the previous split)
X_attention_train, X_attention_test, _, _ = train_test_split(X_attention, y, test_size=0.1, random_state=42)

# Build your model with BERT embeddings
input_ids = tf.keras.Input(shape=(max_length,), dtype=tf.int32)
attention_mask = tf.keras.Input(shape=(max_length,), dtype=tf.int32)

bert_model = TFRobertaModel.from_pretrained('roberta-base')
bert_output = bert_model(input_ids, attention_mask=attention_mask)[0]
output = tf.keras.layers.Dense(len(label_encoder.classes_), activation='softmax')(bert_output[:, 0, :])

model = tf.keras.Model(inputs=[input_ids, attention_mask], outputs=output)

# Compile and train the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit([X_train, X_attention_train], y_train,
                    epochs=47, batch_size=128,
                    validation_data=([X_test, X_attention_test], y_test))