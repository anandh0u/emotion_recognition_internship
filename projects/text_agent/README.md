# Text Agent

Emotion recognition from typed text or speech transcripts.

## Data

Manifest:

```text
E:\emotion_recognition_data\agents\text\manifests\labels_text.csv
```

Current status: workspace ready, but no labeled text dataset has been added yet.

Recommended datasets:

- GoEmotions for text-only emotion
- MELD for dialogue/video/audio/text emotion
- Transcripts generated from speech-to-text, after manual label review

## Runtime

The Streamlit app can already run a text emotion agent:

- Transformer: `j-hartmann/emotion-english-distilroberta-base`
- Fallback: built-in lexical emotion scorer

