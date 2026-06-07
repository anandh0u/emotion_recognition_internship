# Official Agent Workspace Report

Workspace root: `E:\emotion_recognition_data\agents`
Supervisor registry: `E:\emotion_recognition_data\agents\supervisor\agent_registry.json`

## Audio Agent

- Rows: 12162
- By dataset: {'crema_d': 7442, 'ravdess': 1440, 'savee': 480, 'tess': 2800}
- By label: {'anger': 1923, 'disgust': 1923, 'fear': 1923, 'happiness': 1923, 'neutral': 1895, 'sadness': 1923, 'surprise': 652}

## Vision Agent

- Rows: 3360
- By dataset: {'fer2013': 480, 'ravdess_frames': 2880}
- By label: {'anger': 444, 'disgust': 444, 'fear': 444, 'happiness': 444, 'neutral': 696, 'sadness': 444, 'surprise': 444}

## Text Agent

- Rows: 0
- Status: ready for GoEmotions, MELD text, or transcript-labeled rows.

## Animation Agent

- Rows: 1258
- By label: {'not_optimized': 620, 'optimized': 638}
- Status: separate binary animated-content task, not mixed with 7-class emotion labels.

## Multimodal/Fusion Agent

- RAVDESS audio-video manifest is stored under the multimodal workspace.
- SAVEE+FER paired manifest is stored for the original late-fusion experiment.

## Supervisor Agent

- Uses the registry to locate replaceable audio, vision, text, animation, and fusion agents.
