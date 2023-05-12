import os
import torchaudio
import pandas as pd
from parkinsonClassification.audio import AudioUtil
import torch

# Funciones para la aumentación de datos

# ----------------------------
# Esta función genera los nuevos path de guardado, donde se guardarán los
# datos aumentados. Si no existen los crea.
# ----------------------------
def make_dirs(save_path):
    dir_princ = 'aug'
    subdir1 = 'controlAug'
    subdir2 = 'patologicasAug'

    if os.path.exists(os.path.join(save_path, dir_princ)):
        print('Los directorios ya existen')
    else:
        os.mkdir(os.path.join(save_path, dir_princ))
        os.mkdir(os.path.join(save_path, dir_princ, subdir1))
        os.mkdir(os.path.join(save_path, dir_princ, subdir2))


# ----------------------------
# Saca (duracionDelAudio / step) porciones de longitud (lapse)
# ----------------------------
def sliding_window(audio_path, data_path='', lapse=0.5, step=0.1):
    aud = AudioUtil.open(audio_path)
    resamp = AudioUtil.resample(aud, 44100)
    rechan = AudioUtil.rechannel(resamp, 2)
    sig, sr = AudioUtil.pad_trunc(rechan, 2000)

    _, sig_len = sig.shape

    sig_available = (sig_len / sr) - lapse  # Cuanta señal nos queda para dar 'pasos'

    num_steps = int((sig_available // step) + 1)

    start = 0
    names = []
    for i in range(num_steps):  # Se recorta la señal y se almacenan los frames en un vector
        window_begin = int(start * sr)
        window_end = int((start + lapse) * sr)

        window = sig[:, window_begin:window_end]  # Recortamos

        # splitted_path = audio_path.split('\\') # La doble barra es windows. Cuidado Linux
        splitted_path = audio_path.split('/')  # Efectivamente
        new_name = splitted_path[-1].split('.')[0] + f'w{i + 1}' + '.wav'  # Generamos nombre
        copy_spl = splitted_path

        if 'patologicas' in copy_spl:
            add = ['aug', 'patologicasAug', new_name]
            # print(splitted_path)
            # print(add)
        else:
            add = ['aug', 'controlAug', new_name]
            # print(splitted_path)
            # print(add)

        new_path = os.path.join(data_path, *add)
        # print(new_path)

        torchaudio.save(new_path, window, sr)  # Guardamos el audio

        start = start + step
        names.append(new_path)

    # Guardamos la última ventana
    final_window = sig[:, int(start * sr):]

    splitted_path = audio_path.split('/')  # La doble barra es windows. Cuidado Linux
    copy_spl = splitted_path
    new_name = splitted_path[-1].split('.')[0] + f'w{num_steps + 1}' + '.wav'

    if 'patologicas' in copy_spl:
        add = ['aug', 'patologicasAug', new_name]
    else:
        add = ['aug', 'controlAug', new_name]

    new_path = os.path.join(data_path, *add)

    torchaudio.save(new_path, final_window, sr)
    names.append(new_path)

    return names


# ----------------------------
# Esta función recorre el dataset y hace dos cosas: recortar audio y guardar
# cada recorte (llamada a sliding_window) y guardar el path de los recortes.
# ----------------------------
def slide_and_save(df, data_path):
    new_df = pd.DataFrame(columns=['relative_path', 'classID', 'sex'])
    print('Aumentando datos ...')
    for row in df.iloc:
        # Path absoluto del archivo de audio - concatena la carpeta del audio con el
        # path relativo

        audio_file = os.path.join(data_path, row['relative_path'][1:])

        # Toma el ID de la clase
        class_id = row['classID']
        # Toma el género del paciente
        gender = row['sex']
        # Crea las ventanas
        windows = sliding_window(audio_file, data_path)

        # Almacenamos en el nuevo dataset
        for window in windows:
            if (class_id == 0):  # Necesario que el ID sea siempre binario
                splitted_win = window.split('/')  # El path del audio llega entero
                relpath = os.path.join('/', *[splitted_win[-2], splitted_win[-1]])
            else:
                splitted_win = window.split('/')  # El path del audio llega entero
                relpath = os.path.join('/', *[splitted_win[-2], splitted_win[-1]])

            app = pd.Series({'relative_path': relpath, 'classID': class_id})
            new_df = pd.concat([new_df, app.to_frame().T], ignore_index=True).drop_duplicates(keep='first')

    print('Datos aumentados.')
    return new_df

def numpy_to_torch(aud):
    sig = aud[0]
    sigTen = torch.empty((1, len(sig)))
    sigTen[0] = torch.from_numpy(sig)

    return (sigTen, aud[1])

def torch_to_numpy(aud):
    sig = aud[0]
    sigArr = sig.detach().cpu().numpy().squeeze()

    return (sigArr, aud[1])

