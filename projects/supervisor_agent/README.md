# Supervisor Agent

Central coordinator that compares replaceable modality agents.

## Registry

```text
E:\emotion_recognition_data\agents\supervisor\agent_registry.json
```

## Current Agents

- Audio Agent
- Vision Agent
- Text Agent
- Fusion Agent
- Animation Agent as a separate task agent

## Runtime

The Streamlit app uses the supervisor in recommended mode:

```text
audio + image/video + text -> modality agents -> supervisor -> final answer
```

