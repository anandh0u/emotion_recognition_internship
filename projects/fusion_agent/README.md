# Fusion Agent

Combines outputs or embeddings from audio and vision.

## Data

Manifests:

```text
E:\emotion_recognition_data\agents\multimodal\manifests\labels_ravdess_audio_video.csv
E:\emotion_recognition_data\agents\multimodal\manifests\labels_savee_fer_paired.csv
```

## Current Model

The existing late-fusion model projects:

- audio embedding: 768 -> 256
- visual embedding: 768 -> 256
- concat: 512
- classifier: FC -> LayerNorm -> GELU -> Dropout -> classes

## Next Build Step

After the audio and vision agents improve independently, retrain fusion using their stronger embeddings or predictions.

