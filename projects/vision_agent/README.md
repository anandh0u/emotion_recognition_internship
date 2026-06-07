# Vision Agent

Emotion recognition from images and extracted video frames.

## Data

Manifest:

```text
E:\emotion_recognition_data\agents\vision\manifests\labels_vision.csv
```

Current sources:

- FER2013 image rows
- RAVDESS extracted video frames

## Next Build Step

The current visual branch is weak because it uses raw frames/images. The official next step is face detection/cropping before ViT training.

Planned outputs:

```text
E:\emotion_recognition_data\agents\vision\features\
E:\emotion_recognition_data\agents\vision\models\
E:\emotion_recognition_data\agents\vision\results\
```

