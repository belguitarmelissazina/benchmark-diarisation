# Pipelines Diarisation + Transcription — Commandes

## Setup initial (une seule fois)

```powershell
# Création du venv (Python 3.11)
py -3.11 -m venv diarisation
.\diarisation\Scripts\Activate.ps1

# Installation de toutes les dépendances
pip install --upgrade pip
pip install webrtcvad-wheels librosa scikit-learn soundfile numpy `
            imageio-ffmpeg vosk sherpa-onnx faster-whisper diarize `
            speechbrain torch torchaudio silero-vad
```

À chaque nouvelle session :
```powershell
.\diarisation\Scripts\Activate.ps1
```

---

## Pipelines de DIARISATION (qui parle quand → RTTM)

### 1. `diarize_bic_gmm.py` — classique (BIC + GMM, sans deep learning)

Léger, rapide, mais qualité limitée.

```powershell
# Auto
python diarize_bic_gmm.py -i dicte_audio_3.m4a

# Avec nombre de locuteurs forcé (recommandé)
python diarize_bic_gmm.py -i dicte_audio_3.m4a --num-speakers 6

# Désactiver le raffinement GMM
python diarize_bic_gmm.py -i dicte_audio_3.m4a --num-speakers 6 --no-refine

# Tuning de la sensibilité
python diarize_bic_gmm.py -i dicte_audio_3.m4a --bic-threshold -2.0 --bic-lambda-seg 1.0
```

Sortie : `outputs/diarize_bic_gmm/dicte_audio_3/`

---

### 2. `diarize_neural.py` — package `diarize` (FoxNoseTech, WeSpeaker + spectral)

Bonne qualité, ~10.8 % DER, CPU only, pas de token HF.

```powershell
# Auto
python diarize_neural.py -i dicte_audio_3.m4a

# Avec nombre exact
python diarize_neural.py -i dicte_audio_3.m4a --num-speakers 6

# Avec fourchette
python diarize_neural.py -i dicte_audio_3.m4a --min-speakers 4 --max-speakers 8
```

Sortie : `outputs/diarize_neural/dicte_audio_3/`

---

### 3. `diarize_simple.py` — SpeechBrain ECAPA-TDNN (le plus paramétrable)

Embeddings ECAPA-TDNN ou X-Vector + clustering spectral ou hiérarchique.

```powershell
# ECAPA + Spectral Clustering (recommandé)
python diarize_simple.py -i dicte_audio_3.m4a --embed ecapa --cluster sc --num-speakers 6

# Fenêtres plus longues = embeddings plus stables = moins de fragmentation
python diarize_simple.py -i dicte_audio_3.m4a --num-speakers 6 --win-len 2.0 --hop-len 1.0

# AHC avec seuil (auto-détection nb de locuteurs)
python diarize_simple.py -i dicte_audio_3.m4a --cluster ahc --threshold 0.5

# X-Vector au lieu d'ECAPA
python diarize_simple.py -i dicte_audio_3.m4a --embed xvec --cluster sc --num-speakers 6
```

Sortie : `outputs/diarize_simple_ecapa_sc/dicte_audio_3/` (le nom du dossier inclut la config)

---

## Pipelines de TRANSCRIPTION (audio + RTTM → texte par locuteur)

### 1. `transcribe_sherpa.py` — sherpa-onnx kroko (le plus rapide)

Streaming Zipformer français, RTF ~0.1x.

```powershell
# Avec RTTM de diarize_neural (défaut)
python transcribe_sherpa.py -i dicte_audio_3.m4a

# Avec un autre pipeline de diarisation
python transcribe_sherpa.py -i dicte_audio_3.m4a --diar diarize_bic_gmm
python transcribe_sherpa.py -i dicte_audio_3.m4a --diar diarize_simple_ecapa_sc

# Avec un RTTM explicite
python transcribe_sherpa.py -i dicte_audio_3.m4a --rttm chemin/vers/fichier.rttm
```

Sortie : `outputs/transcribe_sherpa/<diar>/dicte_audio_3/`

---

### 2. `transcribe_whisper.py` — faster-whisper (meilleure qualité)

Whisper avec timestamps mot par mot, RTF ~0.3-0.5x avec `small`.

```powershell
# Modèle small (défaut, bon compromis)
python transcribe_whisper.py -i dicte_audio_3.m4a

# Choisir la taille du modèle
python transcribe_whisper.py -i dicte_audio_3.m4a --model tiny     # rapide
python transcribe_whisper.py -i dicte_audio_3.m4a --model small    # défaut
python transcribe_whisper.py -i dicte_audio_3.m4a --model medium   # +qualité
python transcribe_whisper.py -i dicte_audio_3.m4a --model large-v3 # max

# Avec un autre pipeline de diarisation
python transcribe_whisper.py -i dicte_audio_3.m4a --diar diarize_simple_ecapa_sc
```

Sortie : `outputs/transcribe_whisper/<diar>/dicte_audio_3/`

---

### 3. `transcribe_vosk.py` — Vosk (modèle local)

Utilise le modèle `vosk-model-fr-0.22` du dossier.

```powershell
python transcribe_vosk.py -i dicte_audio_3.m4a

# Avec un autre pipeline de diarisation
python transcribe_vosk.py -i dicte_audio_3.m4a --diar diarize_neural
```

Sortie : `outputs/transcribe_vosk/<diar>/dicte_audio_3/`

---

## Workflow complet recommandé

```powershell
# 1. Diarisation (choisir UN pipeline)
python diarize_simple.py -i dicte_audio_3.m4a --embed ecapa --cluster sc --num-speakers 6

# 2. Transcription (avec le RTTM produit)
python transcribe_sherpa.py -i dicte_audio_3.m4a --diar diarize_simple_ecapa_sc
```

Résultat final dans :
```
outputs/transcribe_sherpa/diarize_simple_ecapa_sc/dicte_audio_3/
    ├── dicte_audio_3.transcript.txt   ← le résultat lisible
    └── dicte_audio_3.words.json       ← debug
```

---

## Comparer plusieurs pipelines sur le même audio

```powershell
# Lancer toutes les diarisations
python diarize_bic_gmm.py     -i dicte_audio_3.m4a --num-speakers 6
python diarize_neural.py      -i dicte_audio_3.m4a --num-speakers 6
python diarize_simple.py      -i dicte_audio_3.m4a --num-speakers 6 --embed ecapa --cluster sc

# Lancer toutes les transcriptions sur chaque RTTM
python transcribe_sherpa.py   -i dicte_audio_3.m4a --diar diarize_neural
python transcribe_sherpa.py   -i dicte_audio_3.m4a --diar diarize_simple_ecapa_sc
python transcribe_whisper.py  -i dicte_audio_3.m4a --diar diarize_simple_ecapa_sc --model small
```

Chaque combinaison crée son propre dossier sous `outputs/`, facile à comparer.

---

## Arborescence des sorties

```
outputs/
├── diarize_bic_gmm/dicte_audio_3/             ← RTTM
├── diarize_neural/dicte_audio_3/              ← RTTM
├── diarize_simple_ecapa_sc/dicte_audio_3/     ← RTTM
│
├── transcribe_sherpa/
│   ├── diarize_neural/dicte_audio_3/          ← transcript
│   ├── diarize_simple_ecapa_sc/dicte_audio_3/
│   └── ...
│
├── transcribe_whisper/
│   └── ...
│
└── transcribe_vosk/
    └── ...
```

---

## Astuces

- **Toujours préciser `--num-speakers`** si tu connais le nombre de locuteurs : qualité bien meilleure
- **Modèle small de Whisper** = bon compromis vitesse/qualité sur CPU
- **Sherpa kroko** = le plus rapide mais qualité légèrement inférieure à Whisper small
- **`diarize_simple` avec `--win-len 2.0`** = moins de fragmentation
- **Lissage automatique** : `transcribe_sherpa.py` applique déjà un lissage du RTTM (fusion des micro-segments) lors de la fusion mots ↔ locuteurs
