import torch
import torchaudio
from torchaudio import transforms
import random


# Esta clase va a procesar cada audio

class AudioUtil():

    # ----------------------------
    # Carga un archivo de audio. Devuelve la señal como tensor y
    # el sample rate
    # ----------------------------
    @staticmethod
    def open(audio_file):
        sig, sr = torchaudio.load(audio_file)
        return (sig, sr)

    # ----------------------------
    # Convierte el audio dado al número deseado de canales
    # ----------------------------
    @staticmethod
    def rechannel(aud, new_channel):
        sig, sr = aud

        if (sig.shape[0] == new_channel):
            # Nada que hacer
            return aud

        if (new_channel == 1):
            # Convierte de estéreo a mono usando sólo el primer canal
            resig = sig[:1, :]
        else:
            # Convierte de mono a estéreo duplicando el primer canal
            resig = torch.cat([sig, sig])

        return (resig, sr)

    # ----------------------------
    # Resample de cada canal para tener el mismo sample rate
    # ----------------------------
    @staticmethod
    def resample(aud, newsr):
        sig, sr = aud

        if (sr == newsr):
            # Nada que hacer
            return aud

        num_channels = sig.shape[0]
        # Resample primer canal
        resig = torchaudio.transforms.Resample(sr, newsr)(sig[:1, :])
        if (num_channels > 1):
            # Resample segundo canal y unir ambos
            retwo = torchaudio.transforms.Resample(sr, newsr)(sig[1:, :])
            resig = torch.cat([resig, retwo])

        return (resig, newsr)


    # ----------------------------
    # Prolonga o trunca la señal a una longitud fija para tener la misma
    # dimensión
    # ----------------------------
    @staticmethod
    def pad_trunc(aud, max_ms):
        sig, sr = aud
        num_rows, sig_len = sig.shape
        max_len = sr // 1000 * max_ms

        if (sig_len > max_len):
            # Trunca la señal a una longitud fija
            sig = sig[:, :max_len]

        elif (sig_len < max_len):
            # Calcula la longitud del relleno
            pad_begin_len = random.randint(0, max_len - sig_len)
            pad_end_len = max_len - sig_len - pad_begin_len

            # Relleno con 0's
            pad_begin = torch.zeros((num_rows, pad_begin_len))
            pad_end = torch.zeros((num_rows, pad_end_len))

            sig = torch.cat((pad_begin, sig, pad_end), 1)

        return (sig, sr)
    # ----------------------------
    # Desplaza la señal a izqda o dcha un porcentaje. Los valores del final
    # se 'enrrollan' al principio de la señal transformada.
    # ----------------------------
    @staticmethod
    def time_shift(aud, shift_limit):
        sig, sr = aud
        _, sig_len = sig.shape
        shift_amt = int(random.random() * shift_limit * sig_len)
        return (sig.roll(shift_amt), sr)

    # ----------------------------
    # Genera el Espectrograma
    # ----------------------------
    @staticmethod
    def spectro_gram(aud, n_mels=256, n_fft=1024, hop_len=None):
        sig, sr = aud
        top_db = 80

        # spec tiene forma [channel, n_mels, time], donde channel is mono, stereo etc
        spec = transforms.MelSpectrogram(sr, n_fft=n_fft, hop_length=hop_len, n_mels=n_mels)(sig)

        # Convertimos a decibelios
        spec = transforms.AmplitudeToDB(top_db=top_db)(spec)
        return (spec)