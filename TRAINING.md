# Dataset and model training

VoiceGuard includes a small education-only model and dataset subset. The prediction API returns **UNCERTAIN** when the model is missing, the score is below threshold, or audio decoding fails. The included validation score describes only a held-out split from this subset.

## Prepare labelled audio

Place audio files into this folder layout (WAV is recommended; MP3, FLAC, OGG and M4A are also supported):

```text
dataset/
  real/    # authentic human recordings
  fake/    # generated or voice-converted recordings
```

Use a research dataset with explicit real/fake labels and permission to use it. Keep speakers and source recordings disjoint across training and evaluation when the dataset permits it. Avoid putting clips derived from the same utterance in both sets. The included script makes a stratified random 80/20 split, so speaker-disjoint preparation is strongly recommended for a meaningful estimate. At least 10 files per class are required; more diverse examples are essential for useful results.

## Train

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python train.py --data dataset --epochs 15
```

The training script converts mono audio to 16 kHz, extracts 64-bin log-mel spectrograms, trains a small PyTorch CNN, reports held-out metrics, and saves the best validation checkpoint. Filenames that repeat across generators are kept in the same split. Restart the Flask app after replacing a checkpoint. The classifier's scores are not calibrated probabilities and dataset performance does not establish performance on new speakers or generators.

The bundled subset comes from [MLAAD-tiny](https://huggingface.co/datasets/mueller91/MLAAD-tiny), an educational subset of the MLAAD dataset. Synthetic audio carries CC BY-NC 4.0 terms; authentic M-AILABS audio retains its original license. Read the notices in `dataset/licenses/` and the upstream dataset card before sharing or redistributing any audio. The subset has 300 real and 300 fake clips from six English TTS systems.

The app accepts audio up to 25 MB and analyzes at most the first 30 seconds. MFCCs can be explored for extension; the included CNN currently uses log-mel features. For exhibition use, document your dataset, split, class balance, and evaluation results.
