from train import utils, plots, models
import numpy as np
import config

def run(seq_len, return_sequences, should_train, early_fusion=False):

    data = utils.load_data(config.RECOLA_PICKLE_PATH)
    #plots.plot_all(data["arousal_label"], 'AROUSAL')
    #plots.plot_all(data["valence_label"], 'VALENCE')

    if early_fusion:
        if should_train in ('all', 'audio'):
            arousal_concat = np.concatenate([np.array(data["arousal_audio_feature"]), np.array(data["arousal_video_feature"])], axis=1)
            models.train_model(arousal_concat, data["arousal_label"], "lstm", seq_len, texto="Acoustic features openSMILE and eGeMAPS - features_audio/*.arff (88 features) - TESTE MODELO LSTM SAÍDA ÚNICA AROUSAL\n", return_seq=return_sequences)
            models.train_model(arousal_concat, data["valence_label"], "lstm", seq_len, texto="Acoustic features openSMILE and eGeMAPS - features_audio/*.arff (88 features) - TESTE MODELO LSTM SAÍDA ÚNICA AROUSAL\n", return_seq=return_sequences)
        
        if should_train in ('all', 'video'):
            valence_concat = np.concatenate([np.array(data["valence_audio_feature"]), np.array(data["valence_video_feature"])], axis=1)
            models.train_model(valence_concat, data["arousal_label"], "lstm", seq_len, texto="Acoustic features openSMILE and eGeMAPS - features_audio/*.arff (88 features) - TESTE MODELO LSTM SAÍDA ÚNICA VALENCE\n", return_seq=return_sequences)
            models.train_model(valence_concat, data["valence_label"], "lstm", seq_len, texto="Acoustic features openSMILE and eGeMAPS - features_audio/*.arff (88 features) - TESTE MODELO LSTM SAÍDA ÚNICA VALENCE\n", return_seq=return_sequences)
    else:
        if should_train in ('all', 'ecg'):
        # Eletrocardiograma (ECG)- features_ECG/*.arff (19 features)
        ### Arousal
            models.train_model(np.array(data["arousal_ecg"]), data["arousal_label"], "lstm", seq_len, texto="Eletrocardiograma (ECG)- features_ECG/*.arff - TESTE MODELO LSTM SAÍDA ÚNICA AROUSAL\n", return_seq=return_sequences)
        ### Valence
            models.train_model(np.array(data["valence_ecg"]), data["valence_label"], "lstm", seq_len, texto=f"Eletrocardiograma (ECG)- features_ECG/*.arff - TESTE MODELO LSTM SAÍDA ÚNICA VALENCE\n", return_seq=return_sequences)

        # Heart Rate First Order Derivate (HRV) - features_HRHRV/*.arff (10 features)
        ### Arousal
            models.train_model(np.array(data["arousal_hrv"]), data["arousal_label"], "lstm", seq_len, texto="Heart Rate First Order Derivate (HRV) - TESTE MODELO LSTM SAÍDA ÚNICA AROUSAL\n", return_seq=return_sequences)
        ### Valence
            models.train_model(np.array(data["valence_hrv"]), data["valence_label"], "lstm", seq_len, texto=f"Heart Rate First Order Derivate (HRV) - TESTE MODELO LSTM SAÍDA ÚNICA VALENCE\n", return_seq=return_sequences)

    if should_train in ('all', 'audio'):
        # Acoustic features openSMILE and eGeMAPS - features_audio/*.arff (88 features)
        ### Arousal #TO DO: Alterar chave "arousal_audio_feature"/"valence_audio_feature"
        models.train_model(np.array(data["arousal_audio_feature"]), data["arousal_label"], "lstm", seq_len, texto="Acoustic features openSMILE and eGeMAPS - features_audio/*.arff (88 features) - TESTE MODELO LSTM SAÍDA ÚNICA AROUSAL\n", return_seq=return_sequences)

        ### Valence
        models.train_model(np.array(data["valence_audio_feature"]), data["valence_label"], "lstm", seq_len, texto="Acoustic features openSMILE and eGeMAPS - features_audio/*.arff (88 features) - TESTE MODELO LSTM SAÍDA ÚNICA VALENCE\n", return_seq=return_sequences)

        # Mel-frequency cepstral coefficients (MFCCs) - código próprio (30 features)
        ### Arousal
        #luiz data_mfccs = np.reshape(np.array(data["audio_mfccs"]), (-1, np.array(data["audio_mfccs"]).shape[1]*np.array(data["audio_mfccs"]).shape[2])) 
        #luiz models.train_model(np.array(data_mfccs), data["arousal_label"], "lstm", seq_len, texto="Mel-frequency cepstral coefficients (MFCCs) (30 features) - TESTE MODELO LSTM SAÍDA ÚNICA AROUSAL\n", return_seq=return_sequences)
        ### Valence
        #luiz models.train_model(np.array(data_mfccs), data["valence_label"], "lstm", seq_len, texto="Mel-frequency cepstral coefficients (MFCCs) (30 features) - TESTE MODELO LSTM SAÍDA ÚNICA VALENCE\n", return_seq=return_sequences)

        # Mel Spectrograms - código próprio (65 features)
        ### Arousal
        #luiz data_specs = np.reshape(np.array(data["audio_specs"]), (-1, np.array(data["audio_specs"]).shape[1]*np.array(data["audio_specs"]).shape[2])) #flatten
        #luiz models.train_model(np.array(data_specs), data["arousal_label"], "lstm", seq_len, texto="Mel Spectrograms (65 features) - TESTE MODELO LSTM SAÍDA ÚNICA AROUSAL\n", return_seq=return_sequences)
        ### Valence
        #luiz models.train_model(np.array(data_specs), data["valence_label"], "lstm", seq_len, texto="Mel Spectrograms (65 features) - TESTE MODELO LSTM SAÍDA ÚNICA VALENCE\n", return_seq=return_sequences)

        # Mel-frequency cepstral coefficients (MFCCs) - código próprio (30 features)
        ### Arousal
        #luiz models.train_model(np.array(data["audio_mfccs"]), data["arousal_label"], "speech", seq_len, texto="Mel-frequency cepstral coefficients (MFCCs) (30 features) - TESTE MODELO SPEECH SAÍDA ÚNICA AROUSAL\n", return_seq=return_sequences)
        ### Valence
        #luiz models.train_model(np.array(data["audio_mfccs"]), data["valence_label"], "speech", seq_len, texto="Mel-frequency cepstral coefficients (MFCCs) (30 features) - TESTE MODELO SPEECH SAÍDA ÚNICA VALENCE\n", return_seq=return_sequences)

        # Mel Spectrograms - código próprio (65 features)
        ### Arousal
        #luiz models.train_model(np.array(data["audio_specs"]), data["arousal_label"], "speech", seq_len, texto="Mel Spectrograms (65 features) - TESTE MODELO SPEECH SAÍDA ÚNICA AROUSAL\n", return_seq=return_sequences)
        ### Valence
        #luiz models.train_model(np.array(data["audio_specs"]), data["valence_label"], "speech", seq_len, texto="Mel Spectrograms (65 features) - TESTE MODELO SPEECH SAÍDA ÚNICA VALENCE\n", return_seq=return_sequences)

    if should_train in ('all', 'video'):
        # Appearance video features obtained by a PCA from 50k LGBP-TOP features - features_video_appearance/*.arff (168 features)
        ### Arousal
        models.train_model(np.array(data["arousal_video_feature"]), data["arousal_label"], "lstm", seq_len, texto="Appearance video features obtained by a PCA from 50k LGBP-TOP features - features_video_appearance/*.arff (168 features) - TESTE MODELO LSTM SAÍDA ÚNICA AROUSAL\n", return_seq=return_sequences)
        ### Valence
        models.train_model(np.array(data["valence_video_feature"]), data["valence_label"], "lstm", seq_len, texto="Appearance video features obtained by a PCA from 50k LGBP-TOP features - features_video_appearance/*.arff (168 features) - TESTE MODELO LSTM SAÍDA ÚNICA VALENCE\n", return_seq=return_sequences)

        # Geometric video features derived from 49 facial landmarks - features_video_geometric/*.arff (632 features)
        ### Arousal
        #luizmodels.train_model(np.array(data["arousal_geometric_feature"]), data["arousal_label"], "lstm", seq_len, texto="Geometric video features derived from 49 facial landmarks - features_video_geometric/*.arff (632 features) - TESTE MODELO LSTM SAÍDA ÚNICA AROUSAL\n", return_seq=return_sequences)
        ### Valence
        #luizmodels.train_model(np.array(data["valence_geometric_feature"]), data["valence_label"], "lstm", seq_len, texto="Geometric video features derived from 49 facial landmarks - features_video_geometric/*.arff (632 features) - TESTE MODELO LSTM SAÍDA ÚNICA VALENCE\n", return_seq=return_sequences)