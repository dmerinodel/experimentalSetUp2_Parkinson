from parkinsonClassification.audio import AudioUtil
from torch.utils.data import Dataset

class SoundDS(Dataset):
    def __init__(self, df, data_path):
        self.df = df
        self.data_path = str(data_path)
        self.duration = 500
        self.sr = 44100
        self.channel = 2
        self.shift_pct = 0.4

    # ----------------------------
    # Número de items en el dataset
    # ----------------------------
    def __len__(self):
        return len(self.df)

    # ----------------------------
    # Toma el i-ésimo elemento del dataset
    # ----------------------------
    def __getitem__(self, idx):
        # Path absoluto del archivo de audio - concatena la carpeta del audio con el
        # path relativo
        audio_file = self.data_path + self.df.loc[idx, 'relative_path']
        # Toma el ID de la clase
        class_id = self.df.loc[idx, 'classID']

        """ Si se quiere añadir sexo, descomentar y cambiar return
        sex = self.df.loc[id,'sex']
        """

        aud = AudioUtil.open(audio_file)

        # Usamos las funciones de la clase anterior para normalizar las características
        # de los audios

        reaud = AudioUtil.resample(aud, self.sr)
        rechan = AudioUtil.rechannel(reaud, self.channel)

        dur_aud = AudioUtil.pad_trunc(rechan, self.duration)
        shift_aud = AudioUtil.time_shift(dur_aud, self.shift_pct)
        sgram = AudioUtil.spectro_gram(shift_aud)

        return sgram, class_id

        # return (sgram, sex), class_id
