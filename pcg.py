import pandas as pd
import os
import torch
from torch import nn, optim
import time
import matplotlib.pyplot as plt

from parkinsonClassification.flow import train
from parkinsonClassification.dataset import SoundDS
from parkinsonClassification.models import Model1, count_parameters
from torch.utils.data import DataLoader
import parkinsonClassification.dataug as aug

# Antes de nada nos vamos a quedar con los nombres de los pacientes que van en
# train-test-val
excel_path = 'datasets/PCG_Parkinson_ds/PCGITA_metadata.xlsx'
names = pd.read_excel(excel_path)
names = names[['RECORDING ORIGINAL NAME', 'SEX']]
names['classID'] = [0 if 'C' in name else 1 for name in names['RECORDING ORIGINAL NAME']]

# Primero vamos a separar en hombres y mujeres
names_m = names[names['SEX'] == 'M']
names_f = names[names['SEX'] == 'F']

# Ahora vamos a extraer la muestra estratificada. Muestreamos para 90-10
# Primero nombres de entrenamiento
train_names_m = names_m.groupby('classID', group_keys=False).apply(lambda x: x.sample(frac=0.9))
train_names_f = names_f.groupby('classID', group_keys=False).apply(lambda x: x.sample(frac=0.9))
train_names = pd.concat([train_names_m, train_names_f], ignore_index=True)

# Comprobaciones
tms = len(train_names[(train_names['SEX'] == 'M') & (train_names['classID'] == 0)])
tme = len(train_names[(train_names['SEX'] == 'M') & (train_names['classID'] == 1)])
tfs = len(train_names[(train_names['SEX'] == 'F') & (train_names['classID'] == 0)])
tfe = len(train_names[(train_names['SEX'] == 'F') & (train_names['classID'] == 1)])

# Eliminamos del total
names_m = pd.concat([names_m, train_names_m]).drop_duplicates(keep=False)
names_f = pd.concat([names_f, train_names_f]).drop_duplicates(keep=False)

# Ahora nombres de validación
val_names = pd.concat([names_m, names_f], ignore_index=True)

# Comprobaciones
vms = len(val_names[(val_names['SEX'] == 'M') & (val_names['classID'] == 0)])
vme = len(val_names[(val_names['SEX'] == 'M') & (val_names['classID'] == 1)])
vfs = len(val_names[(val_names['SEX'] == 'F') & (val_names['classID'] == 0)])
vfe = len(val_names[(val_names['SEX'] == 'F') & (val_names['classID'] == 1)])

print('Train: ', len(train_names))
print(f'Hombres sanos: {tms} - Hombres enfermos {tme}')
print(f'Mujeres sanas: {tfs} - Mujeres enfermas {tfe}')
print('Val: ', len(val_names))
print(f'Hombres sanos: {vms} - Hombres enfermos {vme}')
print(f'Mujeres sanas: {vfs} - Mujeres enfermas {vfe}')

# ----------------------------
# Preparando datos de entrenamiento desde los metadatos
# ----------------------------
download_path = 'datasets/PCG_Parkinson_ds/'

# Leemos archivo de metadatos
metadata_file = os.path.join(download_path, 'metadata/PCGITA_metadata.csv')
df = pd.read_csv(metadata_file)

# Construimos el path de los archivos añadiendo el nombre de las carpetas
df['relative_path'] = '/' + df['fold'].astype(str) + '/' + df['RECORDING_ORIGINAL_NAME'].astype(str) + '.wav'

# Nos quedamos con las columnas que importan
df = df[['relative_path', 'classID', 'sex']]

# Ahora vamos a quedarnos con las grabaciones correspondientes a los nombres
# seleccionados en train, val y test.

df_train = pd.DataFrame()
for name in train_names['RECORDING ORIGINAL NAME']:
    df_train = pd.concat([df_train, df[df['relative_path'].str.contains(name)]], ignore_index=True)

df_val = pd.DataFrame()
for name in val_names['RECORDING ORIGINAL NAME']:
    df_val = pd.concat([df_val, df[df['relative_path'].str.contains(name)]], ignore_index=True)

# Comprobaciones.
print('Datos entrenamiento: ', len(df_train))
print('Sanos: ', len(df_train[df_train['classID'] == 0]))
print('Enfermos: ', len(df_train[df_train['classID'] == 1]))
print('-------------------------------------------')
print('Datos validación: ', len(df_val))
print('Sanos: ', len(df_val[df_val['classID'] == 0]))
print('Enfermos: ', len(df_val[df_val['classID'] == 1]))

data_path = 'datasets/PCG_Parkinson_ds/A'

train_ds = SoundDS(df_train, data_path)
val_ds = SoundDS(df_val, data_path)

print('INPUT: ', train_ds.__getitem__(0)[0].shape)

# Cargamos los datos
bs = 64
train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=bs, shuffle=True)
dataloaders = {'train': train_dl, 'valid': val_dl}

# Modelo
myModel = Model1()
# Transfer learning TODO: definir con flag y pasar por parámetro
weights = torch.load('models/pretrain.pt')
myModel.load_state_dict(weights)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
myModel = myModel.to(device)
print(next(myModel.parameters()).device)
print(f'El modelo tiene {count_parameters(myModel)} parámetros')

# Hiperparámetros
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(myModel.parameters(), lr=2e-5)

# Entrenamiento
n_epochs = 100
use_cuda = torch.cuda.is_available()
print('GPU:', use_cuda)
inicio = time.time()
loss_dict, model = train(n_epochs=n_epochs,
                         dataloaders=dataloaders,
                         model=myModel,
                         criterion=criterion,
                         use_cuda=use_cuda,
                         optimizer=optimizer)
final = time.time()
train_time = (final-inicio)/60
print(f'Ha tardado {train_time} minutos.')

fig = plt.figure(figsize=(16, 9))
plt.plot(loss_dict["valid"])
plt.plot(loss_dict["train"])
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend(["valid loss", "training loss"])
plt.title('Model1')
plt.savefig("figures/pcg_loss.png")
plt.close()

fig = plt.figure(figsize=(16, 9))
plt.plot(loss_dict["valid_acc"])
plt.plot(loss_dict["train_acc"])
plt.xlabel("Epoch")
plt.ylabel("Acc")
plt.legend(["valid accuracy", "train accuracy"])
plt.title('Model1')
plt.savefig("figures/pcg_acc.png")
plt.close()

res = open('results/pcg.txt', 'a')
res.write(f'Tiempo Model1: {train_time} minutos\n')
res.write('train_loss \t valid_loss \t train_acc \t valid_acc\n')
for dat1, dat2, dat3, dat4 in zip(loss_dict['train'], loss_dict['valid'],
                                  loss_dict['train_acc'], loss_dict['valid_acc']):
    res.write(f'{dat1} \t {dat2} \t {dat3} \t {dat4}\n')
res.close()
