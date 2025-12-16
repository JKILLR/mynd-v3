# Meta MYND v3.30 - self-dev.html Structure

## Overview

`self-dev.html` is a single-file web application (~43,600 lines) containing all CSS, HTML, and JavaScript for the Meta MYND Self Developer Mode.

---

## File Layout

```
Line 1-15        │ DOCTYPE, <head>, meta tags, external scripts
Line 16-6,980    │ <style> - All CSS
Line 6,981-7,500 │ <body> - HTML elements
Line 7,501-43,600│ <script> - All JavaScript
```

---

## Core Technologies

| Layer | Technology | Purpose |
|-------|------------|---------|
| **3D Rendering** | Three.js | Mind map visualization, nodes, connections |
| **UI Framework** | Vanilla JS + CSS | Panels, modals, themes |
| **AI/ML** | TensorFlow.js | Neural network, embeddings, predictions |
| **Backend** | Supabase | Auth, cloud sync, storage |
| **Storage** | LocalStorage + IndexedDB | Offline data persistence |

---

## CSS Section (~7,000 lines)

| Lines | Section |
|-------|---------|
| 16-400 | CSS Variables (themes, colors, spacing) |
| 400-800 | Base styles, typography, utilities |
| 800-1,500 | Layout (header, panels, canvas) |
| 1,500-2,500 | Components (buttons, inputs, modals) |
| 2,500-3,500 | Side panel & settings |
| 3,500-4,500 | Chat interface |
| 4,500-5,500 | Mobile responsive styles |
| 5,500-6,500 | Animations & transitions |
| 6,500-6,980 | Theme variations |

---

## HTML Section (~500 lines)

| Element | Purpose |
|---------|---------|
| `#loading` | Initial loading spinner |
| `#header` | Top brand bar |
| `#mobile-header` | Mobile navigation |
| `#canvas-container` | Three.js render target |
| `#side-panel` | Node details & editing |
| `#chat-panel` | AI chat interface |
| `#settings-modal` | Settings overlay |
| `#auth-modal` | Login/signup |
| `#version-indicator` | Version badge |

---

## JavaScript Section (~36,000 lines)

| Lines | System | Description |
|-------|--------|-------------|
| 9,160-9,200 | **Config & Globals** | SELF_DEVELOPER_MODE flag, storage keys |
| 9,200-9,650 | **Supabase Init** | Auth client setup |
| 9,650-10,500 | **TensorFlow Loader** | Lazy load ML libraries |
| 10,500-12,000 | **Store Class** | Mind map data CRUD, import/export |
| 12,000-14,500 | **PersonalNeuralNet** | Neural network for suggestions |
| 14,500-16,000 | **FeedbackLearner** | Learns from user actions |
| 16,000-18,000 | **SemanticEngine** | Text embeddings (USE model) |
| 18,000-19,500 | **UserProfile** | Behavior tracking |
| 19,500-21,500 | **MetaLearner** | Adaptive learning |
| 21,500-23,500 | **SelfImprover** | Code self-improvement |
| 23,500-25,500 | **AutonomousEvolution** | Self-dialogue system |
| 25,500-27,500 | **VisionCore** | Mission/purpose system |
| 27,500-29,000 | **Data Export/Import** | Full JSON with neural data |
| 29,000-31,000 | **Three.js Scene** | initScene(), createNode(), render() |
| 31,000-33,500 | **Node Interactions** | Click, drag, select, edit |
| 33,500-35,500 | **Chat Manager** | AI chat, voice input |
| 35,500-37,500 | **UI Handlers** | Menus, panels, modals |
| 37,500-38,800 | **Auth & Sync** | Supabase auth, cloud sync |
| 38,800-39,100 | **Init & Startup** | init() function, app bootstrap |

---

## Major Classes (21 Total)

| Class | Purpose |
|-------|---------|
| `Store` | Central data management for mind map nodes |
| `PersonalNeuralNet` | Learns user patterns, suggests nodes |
| `SemanticMemory` | Stores context and relationships |
| `UserProfile` | Tracks user behavior/preferences |
| `MetaLearner` | Adapts AI suggestions over time |
| `CognitiveGNN` | Graph neural network for node relationships |
| `SemanticEngine` | Text embeddings and similarity |
| `CodeRAG` | Code analysis and retrieval |
| `SelfImprover` | Self-improvement engine for code patches |
| `AutonomousEvolution` | Self-dialogue and recursive improvement |
| `VisionCore` | Foundational purpose/mission system |
| `PreferenceTracker` | Tracks user preferences |
| `RelationshipClassifier` | Classifies node relationships |
| `ConceptAbstractor` | Abstracts concepts from nodes |
| `CognitiveGraphTransformer` | Transformer for graph data |
| `CognitiveStyleFingerprint` | User style fingerprinting |
| `UniversalPatternLibrary` | Stores common patterns |
| `TransferEngine` | Transfers learning between contexts |
| `StyleTransferSystem` | Transfers user styles |
| `WebGPUCompute` | GPU acceleration (when available) |
| `EventBus` | Event pub/sub system |

---

## Key Global Variables

```javascript
// Core
let store;              // Mind map data
let scene, camera, renderer, controls;  // Three.js
let selectedNode;       // Currently selected node
let nodeMeshes = {};    // Node ID → Three.js mesh

// AI/ML
let neuralNet;          // PersonalNeuralNet instance
let semanticEngine;     // SemanticEngine instance
let metaLearner;        // MetaLearner instance
let userProfile;        // UserProfile instance

// Config
const SELF_DEVELOPER_MODE = true;  // Enables self-dev features
const SELF_DEV_STORAGE_KEY = 'mynd_self_dev_data';
```

---

## Data Flow

```
┌─────────────────────────────────────────────────────────┐
│                    self-dev.html                         │
├─────────────────────────────────────────────────────────┤
│  User Action                                             │
│       ↓                                                  │
│  Store.addNode() / Store.updateNode()                   │
│       ↓                                                  │
│  LocalStorage (persist)                                  │
│       ↓                                                  │
│  buildScene() → Three.js renders nodes                  │
│       ↓                                                  │
│  NeuralNet.train() → learns patterns                    │
│       ↓                                                  │
│  SemanticEngine → generates embeddings                  │
│       ↓                                                  │
│  AI Suggestions appear in UI                            │
└─────────────────────────────────────────────────────────┘
```

---

## External Dependencies

| Dependency | Source | Purpose |
|------------|--------|---------|
| Three.js v0.160.0 | CDN | 3D graphics |
| TensorFlow.js v4.17.0 | CDN (lazy) | ML inference |
| Universal Sentence Encoder v1.3.3 | CDN (lazy) | Text embeddings |
| Supabase SDK v2 | CDN | Backend services |
| `js/config.js` | Local | Configuration |
| `js/goal-system.js` | Local | Goal tracking module |

---

## Self Developer Mode Features

When `SELF_DEVELOPER_MODE = true`:

1. **Onboarding Skipped** - Loads directly into app
2. **Local Storage Only** - Uses `mynd_self_dev_data` key
3. **Cloud Sync Disabled** - Works offline
4. **Self-Improvement Engine** - Can analyze and modify its own code
5. **Vision Document** - Editable mission/purpose system
6. **File System Storage** - Can save to local files (Chrome/Edge)

---

*Generated for Meta MYND v3.30*
