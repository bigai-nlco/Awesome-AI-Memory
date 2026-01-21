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
- [Memos: A memory os for ai system](https://arxiv.org/abs/2507.03724)
### 1.1 Memory Analysis of Transformers  
#### 1.1.1 Knowledge Memorization
- [Dissecting recall of factual associations in auto-regressive language models](https://aclanthology.org/2023.emnlp-main.751/)
- [Knowledge circuits in pretrained transformers](https://dl.acm.org/doi/10.5555/3737916.3741681)
- [Interpreting key mechanisms of factual recall in transformer-based language models](https://arxiv.org/abs/2403.19521)
- [Confidence regulation neurons in language models](https://dl.acm.org/doi/10.5555/3737916.3741886)
- [Revealing the dark secrets of BERT](https://aclanthology.org/D19-1445/)
- [Transformer Feed-Forward Layers Are Key-Value Memories](https://aclanthology.org/2021.emnlp-main.446/)
- [Transformer feed-forward layers build predictions by promoting concepts in the vocabulary space](https://aclanthology.org/2022.emnlp-main.3/)
- [Knowledge Neurons in Pretrained Transformers](https://aclanthology.org/2022.acl-long.581/)
- [What does bert look at? an analysis of bert’s attention](https://aclanthology.org/W19-4828/)
- [exBERT: A Visual Analysis Tool to Explore Learned Representations in Transformer Models](https://aclanthology.org/2020.acl-demos.22/)
- [Characterizing mechanisms for factual recall in language models](https://aclanthology.org/2023.emnlp-main.615/)
##### Knowledge is encoded in FFNs.
- [Transformer Feed-Forward Layers Are Key-Value Memories](https://aclanthology.org/2021.emnlp-main.446/)
- [Transformer feed-forward layers build predictions by promoting concepts in the vocabulary space](https://aclanthology.org/2022.emnlp-main.3/)
- [Empirical study on updating key-value memories in transformer feed-forward layers](https://arxiv.org/abs/2402.12233)
- [Understanding transformer from the perspective of associative memory](https://arxiv.org/abs/2505.19488)
- [Knowledge Neurons in Pretrained Transformers](https://aclanthology.org/2022.acl-long.581/)
- [MoEfication: Transformer feed-forward layers are mixtures of experts.](https://aclanthology.org/2022.findings-acl.71)
- [Emergent modularity in pre-trained transformers](https://aclanthology.org/2023.findings-acl.250)
- [Depn: Detecting and editing privacy neurons in pretrained language models](https://aclanthology.org/2023.emnlp-main.174/)
- [Confidence regulation neurons in language models](https://dl.acm.org/doi/10.5555/3737916.3741886)
##### Attention is More Crucial for Knowledge Storage
- [What does bert look at? an analysis of bert’s attention](https://aclanthology.org/W19-4828/)
- [Characterizing mechanisms for factual recall in language models](https://aclanthology.org/2023.emnlp-main.615/)
- [Inference-time intervention: Eliciting truthful answers from a language model](https://dl.acm.org/doi/10.5555/3666122.3667919)
- [Stanford alpaca: An instruction-following llama model](https://crfm.stanford.edu/2023/03/13/alpaca.html)
- [Do llms dream of elephants (when told not to)? latent concept association and associative memory in transformers](https://dl.acm.org/doi/10.5555/3737916.3740079)
##### Knowledge Flows in Connections
- [Knowledge circuits in pretrained transformers](https://dl.acm.org/doi/10.5555/3737916.3741681)
- [Dissecting recall of factual associations in auto-regressive language models](https://aclanthology.org/2023.emnlp-main.751/)
- [Interpreting key mechanisms of factual recall in transformer-based language models](https://arxiv.org/abs/2403.19521)
##### Scaling Law for Knowledge Memorization
- [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361)
- [Scaling Laws for Fact Memorization of Large Language Models](https://aclanthology.org/2024.findings-emnlp.658/)
- [Physics of Language Models: Part 3.3, Knowledge Capacity Scaling Laws](https://arxiv.org/abs/2404.05405)
- [Physics of language models: Part 3.1, knowledge storage and extraction]()
#### 1.1.2 Associative Memory
- [Content-Addressable and Associative Memory](https://doi.org/10.1016/S0065-2458(08)60326-5)
##### Energy-based Model Mimic Associative Memory
- [Neural networks and physical systems with emergent collective computational abilities](https://www.pnas.org/doi/abs/10.1073/pnas.79.8.2554)
- [Hopfield networks is all you need](https://arxiv.org/abs/2008.02217)
##### Transformer-based Models
- [Birth of a transformer: a memory viewpoint](https://dl.acm.org/doi/10.5555/3666122.3666199)
- [Do llms dream of elephants (when told not to)? latent concept association and associative memory in transformers](https://dl.acm.org/doi/10.5555/3737916.3740079)
##### Scaling Law for Associative Memory
- [Scaling Laws for Associative Memories](https://openreview.net/forum?id=Tzh6xAJSll)
- [Beyond Scaling Laws: Understanding Transformer Performance with Associative Memory](https://doi.org/10.48550/arXiv.2405.08707)
##### Usage of Associative Memory
- [CAMELoT: Towards Large Language Models with Training-Free Consolidated Associative Memory](https://arxiv.org/abs/2402.13449)
- [Hierarchical Associative Memory](https://arxiv.org/abs/2107.06446)
- [Universal Hopfield Networks: A General Framework for Single-Shot Associative Memory Models](https://proceedings.mlr.press/v162/millidge22a.html)
### 1.2 Implicit Memory Modification
- [Editing conceptual knowledge for large language models](https://aclanthology.org/2024.findings-emnlp.40/)
- [Knowledge Neurons in Pretrained Transformers](https://aclanthology.org/2022.acl-long.581/)
#### 1.2.1 Modification Methods
##### Incremental Training
- [Modifying memories in transformer models](https://arxiv.org/abs/2012.00363)
- [Propagating Knowledge Updates to LMs Through Distillation](https://dl.acm.org/doi/10.5555/3666122.3668163)
- [Training Language Models with Memory Augmentation](https://aclanthology.org/2022.emnlp-main.382/)
- [RECKONING: reasoning through dynamic knowledge encoding](https://dl.acm.org/doi/10.5555/3666122.3668855)
- [LoRA: Low-Rank Adaptation of Large Language Models](https://dblp.org/rec/journals/corr/abs-2106-09685.html)
- [K-Adapter: Infusing Knowledge into Pre-Trained Models with Adapters](https://aclanthology.org/2021.findings-acl.121/)
- [DEMix Layers: Disentangling Domains for Modular Language Modeling](https://aclanthology.org/2022.naacl-main.407/)
##### Memory Editing
- [Locating and editing factual associations in GPT](https://dl.acm.org/doi/10.5555/3600270.3601532)
- [Mass-editing memory in a transformer](https://arxiv.org/abs/2210.07229)
- [Calibrating Factual Knowledge in Pretrained Language Models](https://aclanthology.org/2022.findings-emnlp.438/)
- [Editing Factual Knowledge in Language Models](https://aclanthology.org/2021.emnlp-main.522/)
- [Methods for measuring, updating, and visualizing factual beliefs in language models](https://aclanthology.org/2023.eacl-main.199/)
- [Large Scale Knowledge Washing](https://arxiv.org/abs/2405.16720)
- [A comprehensive study of knowledge editing for large language models](https://arxiv.org/abs/2401.01286)
##### Memory Unlearning
- [To Forget or Not? Towards Practical Knowledge Unlearning for Large Language Models](https://aclanthology.org/2024.findings-emnlp.82/)
- [Towards Safer Large Language Models through Machine Unlearning](https://aclanthology.org/2024.findings-acl.107/)
- [LLM Surgery: Efficient Knowledge Unlearning and Editing in Large Language Models](https://arxiv.org/abs/2409.13054)
- [Unlearn What You Want to Forget: Efficient Unlearning for LLMs](https://aclanthology.org/2023.emnlp-main.738/)
#### 1.2.2 Modification Benchmark
- [Knowledge editing for large language models: A survey](https://dl.acm.org/doi/10.1145/3698590)
- [Evaluating the Ripple Effects of Knowledge Editing in Language Models](https://aclanthology.org/2024.tacl-1.16/)
- [Cross-Lingual Multi-Hop Knowledge Editing--Benchmarks, Analysis and a Simple Contrastive Learning based Approach](https://aclanthology.org/2024.findings-emnlp.701/)
- [A comprehensive study of knowledge editing for large language models](https://arxiv.org/abs/2401.01286)
- [MQuAKE: Assessing Knowledge Editing in Language Models via Multi-Hop Questions](https://aclanthology.org/2023.emnlp-main.971/)
- [Eva-kellm: A new benchmark for evaluating knowledge editing of llms](https://arxiv.org/abs/2308.09954)



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


-[Memory-Augmented Temporal Dynamic Learning for Action Recognition]()
-[Distinguishing Homophenes Using Multi-Head Visual-Audio Memory for Lip Reading]()
-[SyncTalkFace: Talking Face Generation with Precise Lip-Syncing via Audio-Lip Memory]()
-[MART: Memory-Augmented Recurrent Transformer for Coherent Video Paragraph Captioning]()
-[Heterogeneous Memory Enhanced Multimodal Attention Model for Video Question Answering]()
-[Scene Memory Transformer for Embodied Agents in Long-Horizon Tasks]()
-[STMTrack: Template-Free Visual Tracking With Space-Time Memory Networks]()
-[Memory Oriented Transfer Learning for Semi-Supervised Image Deraining]()
-[Memory-Guided Unsupervised Image-to-Image Translation]()
-[Multi-Scale Memory-Based Video Deblurring]()
-[Video Prediction Recalling Long-Term Motion Context via Memory Alignment Learning]()
-[TrackFormer: Multi-Object Tracking with Transformers]()
-[Attend to You: Personalized Image Captioning with Context Sequence Memory Networks]()
-[Memory-Attended Recurrent Network for Video Captioning]()
-[M3: Multimodal Memory Modelling for Video Captioning]()
-[Remember the Difference: Cross-Domain Few-Shot Semantic Segmentation via Meta-Memory Transfer]()
-[Structured Scene Memory for Vision-Language Navigation]()
-[MeMViT: Memory-Augmented Multiscale Vision Transformer for Efficient Long-Term Video Recognition]()
-[Recurring the Transformer for Video Action Recognition]()
-[Coloring With Limited Data: Few-Shot Colorization via Memory Augmented Networks]()
-[Streaming Video Model]()
-[XMem: Long-Term Video Object Segmentation with an Atkinson-Shiffrin Memory Model]()
-[Video Object Segmentation with Episodic Graph Memory Networks]()
-[MENet: A Memory-Based Network with Dual-Branch for Efficient Event Stream Processing]()
-[Learning Dynamic Memory Networks for Object Tracking]()
-[Video Summarization with Long Short-Term Memory]()
-[Divide-and-Assemble: Learning Block-wise Memory for Unsupervised Anomaly Detection]()
-[Multi-modality Associative Bridging through Memory: Speech Sound Recollected from Face Video]()
-[Visual Graph Memory with Unsupervised Representation for Visual Navigation]()
-[A Read-Write Memory Network for Movie Story Understanding]()
-[Video Object Segmentation Using Space-Time Memory Networks]()
-[Alignment Before Aggregation: Trajectory Memory Retrieval Network for Video Object Segmentation]()
-[EMMN: Emotional Motion Memory Network for Audio-driven Emotional Talking Face Generation]()
-[Learning Video Object Segmentation with Visual Memory](https://doi.org/10.1109/ICCV.2017.480)
-[Memory-and-Anticipation Transformer for Online Action Understanding]()
-[Learning Meta-class Memory for Few-Shot Semantic Segmentation]()
-[MAMo: Leveraging Memory and Attention for Monocular Video Depth Estimation]()
-[Recurrent Filter Learning for Visual Tracking]()
-[Out of Sight, Still in Mind: Reasoning and Planning about Unobserved Objects with Video Tracking Enabled Memory Models]()
-[Feature Augmented Memory with Global Attention Network for VideoQA]()
-[Memory-augmented conformer for improved end-to-end long-form ASR]()
-[Memory-based Semantic Segmentation for Off-road Unstructured Natural Environments]()
-[Memory-Augmented Reinforcement Learning for Image-Goal Navigation]()
-[Local Memory Attention for Fast Video Semantic Segmentation]()
-[Extractive Video Summarizer with Memory Augmented Neural Networks]()
-[Stacked Memory Network for Video Summarization]()
-[Dual Temporal Memory Network for Efficient Video Object Segmentation]()
-[MV-TON: Memory-based Video Virtual Try-on network]()
-[Glance and Focus: Memory Prompting for Multi-Event Video Question Answering]()
-[TransformerFusion: Monocular RGB Scene Reconstruction using Transformers]()
-[Rethinking Space-Time Networks with Improved Memory Coverage for Efficient Video Object Segmentation]()
-[Visual Memory for Robust Path Following]()
-[MultiON: Benchmarking Semantic Map Memory using Multi-Object Navigation]()
-[Long Short-Term Transformer for Online Action Detection]()
-[Associating Objects with Transformers for Video Object Segmentation]()
-[Reading Relevant Feature from Global Representation Memory for Visual Object Tracking]()
-[Universal Face Restoration With Memorized Modulation](https://arxiv.org/abs/2110.01033)
-[Memories are One-to-Many Mapping Alleviators in Talking Face Generation](https://arxiv.org/abs/2212.05005)
-[ChatVideo: A Tracklet-centric Multimodal and Versatile Video Understanding System](https://arxiv.org/abs/2304.14407)
-[Multiscale Memory Comparator Transformer for Few-Shot Video Segmentation](https://arxiv.org/abs/2307.07812)
-[MovieChat: From Dense Token to Sparse Memory for Long Video Understanding](https://arxiv.org/abs/2307.16449)
-[Loop Copilot: Conducting AI Ensembles for Music Generation and Iterative Editing](https://arxiv.org/abs/2310.12404)
-[JARVIS-1: Open-World Multi-task Agents with Memory-Augmented Multimodal Language Models](https://arxiv.org/abs/2311.05997)
-[GPT-4V in Wonderland: Large Multimodal Models for Zero-Shot Smartphone GUI Navigation](https://arxiv.org/abs/2311.07562)
-[Explore, Select, Derive, and Recall: Augmenting LLM with Human-like Memory for Mobile Task Automation](https://arxiv.org/abs/2312.03003)
-[LifelongMemory: Leveraging LLMs for Answering Queries in Egocentric Videos](https://arxiv.org/abs/2312.05269)
-[AppAgent: Multimodal Agents as Smartphone Users](https://arxiv.org/abs/2312.13771)
-[A Simple LLM Framework for Long-Range Video Question-Answering](https://arxiv.org/abs/2312.17235)
-[DoraemonGPT: Toward Understanding Dynamic Scenes with Large Language Models](https://arxiv.org/abs/2401.08392)
-[Multimodal Embodied Interactive Agent for Cafe Scene](https://arxiv.org/abs/2402.00290)
-[Memory Consolidation Enables Long-Context Video Understanding](https://arxiv.org/abs/2402.05861)
-[OS-Copilot: Towards Generalist Computer Agents with Self-Improvement](https://arxiv.org/abs/2402.07456)
-[MemoNav: Working Memory Model for Visual Navigation](https://arxiv.org/abs/2402.19161)
-[MR-MT3: Memory Retaining Multi-Track Music Transcription to Mitigate Instrument Leakage](https://arxiv.org/abs/2403.10024)
-[VideoAgent: Long-form Video Understanding with Large Language Model as Agent](https://arxiv.org/abs/2403.10517)
-[VideoAgent: A Memory-augmented Multimodal Agent for Video Understanding](https://arxiv.org/abs/2403.11481)
-[Language Repository for Long Video Understanding](https://arxiv.org/abs/2403.14622)
-[Efficient Video Object Segmentation via Modulated Cross-Attention Memory](https://arxiv.org/abs/2403.17937)
-[MemFlow: Optical Flow Estimation and Prediction with Memory](https://arxiv.org/abs/2404.04808)
-[MA-LMM: Memory-Augmented Large Multimodal Model for Long-Term Video Understanding](https://arxiv.org/abs/2404.05726)
-[Streaming Long Video Understanding with Large Language Models](https://arxiv.org/abs/2405.16009)
-[Flash-VStream: Memory-Based Real-Time Understanding for Long Video Streams](https://arxiv.org/abs/2406.08085)
-[RMem: Restricted Memory Banks Improve Video Object Segmentation](https://arxiv.org/abs/2406.08476)
-[DrVideo: Document Retrieval Based Long Video Understanding](https://arxiv.org/abs/2406.12846)
-[Deep Visual Odometry With Adaptive Memory]()
-[AAP-MIT: Attentive Atrous Pyramid Network and Memory Incorporated Transformer for Multisentence Video Description]()
-[Learning Recurrent Memory Activation Networks for Visual Tracking]()
-[STAM: A SpatioTemporal Attention Based Memory for Video Prediction]()
-[CroMM-VSR: Cross-Modal Memory Augmented Visual Speech Recognition]()
-[Audio-driven Talking Face Video Generation with Natural Head Pose](https://arxiv.org/abs/2002.10137)
-[SAM 2: Segment Anything in Images and Videos](https://arxiv.org/abs/2408.00714)
-[Online Temporal Action Localization with Memory-Augmented Transformer](https://arxiv.org/abs/2408.02957)
-[OmniDrive: A Holistic LLM-Agent Framework for Autonomous Driving with 3D Perception, Reasoning and Planning](https://arxiv.org/abs/2405.01533)
-[Videollamb: Long-context video understanding with recurrent memory bridges](https://arxiv.org/abs/2409.01071)

### 4.1 Multimodal Context Modeling with Memory
#### 4.1.1 Audio Context Modeling
- [Locating and editing factual associations in GPT](https://dl.acm.org/doi/10.5555/3600270.3601532)
- [Mass-editing memory in a transformer](https://arxiv.org/abs/2210.07229)
- [Calibrating Factual Knowledge in Pretrained Language Models](https://aclanthology.org/2022.findings-emnlp.438/)
- [Editing Factual Knowledge in Language Models](https://aclanthology.org/2021.emnlp-main.522/)
- [Methods for measuring, updating, and visualizing factual beliefs in language models](https://aclanthology.org/2023.eacl-main.199/)
- [Large Scale Knowledge Washing](https://arxiv.org/abs/2405.16720)
- [A comprehensive study of knowledge editing for large language models](https://arxiv.org/abs/2401.01286)

#### 4.1.2 Video Context Modeling
#### 4.1.3 Other Modalities


### 4.2 Downstream tasks




### 4.3 Multimodal Contextual Memory for Robotics
#### 4.3.1 Multimodal Memory-Augmented Agents
#### 4.3.2 Memory-Enhanced Navigation, Odometry, and Manipulation
#### 4.3.3 Application


### 4.4 Limitations, Open Questions, Discussion


