# LLM-Memory-Survey
Memory plays a foundational role in augmenting the reasoning, adaptability, and contextual fidelity of modern Large Language Models (LLMs) and Multi-Modal LLMs (MLLMs). As these models transition from static predictors to interactive systems capable of continual learning and personalized inference, the incorporation of memory mechanisms has emerged as a central theme in their architectural and functional evolution. This survey presents a comprehensive and structured synthesis of memory in LLMs and MLLMs, organizing the literature into a cohesive taxonomy comprising implicit, explicit, and agentic memory paradigms.

## Analogy to Human Brain

The architecture of memory in modern (M)LLMs is increasingly analogous to the synergistic relationship between different human brain systems, particularly the neocortex, the hippocampus, and the prefrontal cortex. This brain-inspired framework, which echoes the principles of Complementary Learning Systems theory, provides a powerful lens through which to understand the different memory paradigms evolving in AI.

1. **Implicit Memory: The Neocortex.**: We conceptualize the model's internal parameters as its digital neocortex. In the brain, the neocortex is the primary repository for long-term semantic knowledge, skills, and consolidated memories, which are learned slowly and stored in a distributed manner. Similarly, a transformer's weights embody the implicit memory of the model, the foundational "world knowledge" acquired during pre-training. This parametric knowledge represents the model's stable, generalized understanding of language, patterns, and facts.

2. **Explicit Memory: The Hippocampal System.**: To access specific, real-time, or episodic information, an AI system requires a mechanism analogous to the hippocampus. The hippocampus is critical for the rapid encoding of new episodic memories (\ie, specific events and their context) and acts as an index that binds together disparate elements of an experience stored across the neocortex. Explicit memory systems in AI, such as Retrieval-Augmented Generation (RAG), mimic this function. They serve as an "AI Hippocampus" by providing an on-demand, queryable index to external information (vector embeddings, knowledge graphs). This allows the model to ground its responses in specific, up-to-date facts without the need for slow, resource-intensive retraining of its entire parametric base (the "neocortex").

3. **Agentic Memory: The Prefrontal Cortex.**: The functionality of agentic memory is best analogized to the prefrontal cortex (PFC), the brain's executive control center. The PFC is responsible for working memory, goal-directed planning, and integrating information from both long-term stores (neocortex) and recent episodic memories (hippocampus) to guide behavior. Agentic memory systems similarly maintain a persistent state across interactions, manage working memory (\eg, a scratchpad), and orchestrate the strategic retrieval and use of both implicit and explicit memory to formulate plans and execute complex tasks. Furthermore, as we explore in xxx, this executive function extends to integrating information from specialized memory modules for spatial, temporal, and embodied intelligence, akin to how the PFC coordinates inputs from various sensory cortices.

🌟🌟🌟  Please feel free to make a PR if I missed something. 🌟🌟🌟 


Table of Contents

- [1 Implicit Memory: Unveiling Knowledge Inside Transformers](#1-implicit-memory-unveiling-knowledge-inside-transformers)
    - [1.1 Memory Analysis of Transformers](#11-memory-analysis-of-transformers)
        - [1.1.1 Knowledge Memorization](#121-video-representation)
        - [1.1.2 Associative Memory](#122-large-video-language-models)
    - [1.2 Implicit Memory Modification](#12-video-context-modeling-with-memory)
        - [1.2.1 Modification Methods](#124-video-caption)
        - [1.2.2 Modification Benchmark](#125-videoqa)
    - [1.3 Limitations, Open Questions, Discussion](#13-other-modality-context-modeling-with-memory)
- [2 Explicit Memory: When (M)LLMs Meet Retrieval](#2-multimodal-contextual-memory-for-vision-and-robotics)
    - [2.1 Explicit Memory Representation](#11-memory-analysis-of-transformers)
        - [2.1.1 Free text](#121-video-representation)
        - [2.1.2 Graph](#122-large-video-language-models)
        - [2.1.3 Vector](#122-large-video-language-models)
    - [2.2 Training with Explicit Memory](#12-video-context-modeling-with-memory)
        - [2.2.1 Pre-Training](#124-video-caption)
        - [2.2.2 Fine-Tuning](#125-videoqa)
    - [2.3 Training with externalized parameteric knowledge](#13-other-modality-context-modeling-with-memory)
        - [2.3.1 Long Contexts](#124-video-caption)
        - [2.3.2 Knowledge Injection](#125-videoqa)
    - [2.4 Limitations, Open Questions, Discussion](#13-other-modality-context-modeling-with-memory)     
- [3 Agentic Memory: Consolidating Memories into Humanic Agents](#3-external-multimodal-knowledgememory-augmentation)
    - [3.1 Single-agent Memory](#31-external-audiospeech-knowledgememory)
        - [3.1.1 Short-term Memory](#311-audio-caption)
        - [3.1.2 Long-term Memory](#312-asr)
    - [3.2 Multi-agent Memory](#32-external-image-knowledgememory)
    - [3.3 System Architecture](#33-external-video-knowledgememory)
        - [3.3.1 Data Ingestion](#331-video-caption)
        - [3.3.2 Storage and Retrieval](#332-videoqa)
        - [3.3.3 User Interfaces and Application Invocation](#332-videoqa)
    - [3.4 Evaluation on Agent Memory](#34-external-3d-knowledgememory)
        - [3.4.1 Qualitative Evaluation](#331-video-caption)
        - [3.4.2 Quantitative Evaluation](#332-videoqa)
    - [3.5 Limitations, Open Questions, Discussion](#35-retrieval-augmented-multimodal-large-language-model)
- [4 Memory-augmented Multi-Modal Large Language models](#3-external-multimodal-knowledgememory-augmentation)
    - [4.1 Multimodal Context Modeling with Memory](#31-external-audiospeech-knowledgememory)
        - [4.1.1 Audio Context Modeling](#311-audio-caption)
        - [4.1.2 Video Context Modeling](#312-asr)
        - [4.1.3 Other Modalities](#312-asr)
    - [4.2 Downstream tasks](#32-external-image-knowledgememory)
    - [4.3 Multimodal Contextual Memory for Robotics](#33-external-video-knowledgememory)
        - [4.3.1 Multimodal Memory-Augmented Agents](#331-video-caption)
        - [4.3.2 Memory-Enhanced Navigation, Odometry, and Manipulation](#332-videoqa)
        - [4.3.3 Application](#332-videoqa)
    - [4.4 Limitations, Open Questions, Discussion](#35-retrieval-augmented-multimodal-large-language-model)




## 1 Implicit Memory: Unveiling Knowledge Inside Transformers


### 1.1 Memory Analysis of Transformers

- xxxx
  
#### 1.1.1 Knowledge Memorization
#### 1.1.2 Associative Memory

### 1.2 Implicit Memory Modification

#### 1.2.1 Modification Methods



#### 1.2.2 Modification Benchmark


### 1.3 Limitations, Open Questions, Discussion



## 2 Explicit Memory: When (M)LLMs Meet Retrieval

### 2.1 Explicit Memory Representation



### 2.2 Training with Explicit Memory




### 2.3 Training with externalized parameteric knowledge



### 2.4 Limitations, Open Questions, Discussion



## 3 Agentic Memory: Consolidating Memories into Humanic Agents
### 3.1 Single-agent Memory



### 3.2 Multi-agent Memory




### 3.3 System Architecture



### 3.4 Evaluation on Agent Memory

### 3.5 Limitations, Open Questions, Discussion



## 4 Memory-augmented Multi-Modal Large Language models

### 4.1 Multimodal Context Modeling with Memory



### 4.2 Downstream tasks




### 4.3 Multimodal Contextual Memory for Robotics



### 4.4 Limitations, Open Questions, Discussion


