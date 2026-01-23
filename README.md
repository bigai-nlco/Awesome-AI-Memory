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
        - [1.1.1 Knowledge Memorization](#111-knowledge-memorization)
        - [1.1.2 Associative Memory](#112-associative-memory)
    - [1.2 Implicit Memory Modification](#12-implicit-memory-modification)
        - [1.2.1 Modification Methods](#121-modification-methods)
        - [1.2.2 Modification Benchmark](#122-modification-benchmark)
    - [1.3 Limitations, Open Questions, Discussion](#13-limitations-open-questions-discussion)
- [2 Explicit Memory: When (M)LLMs Meet Retrieval](#2-explicit-memory-when-mllms-meet-retrieval)
    - [2.1 Explicit Memory Representation](#21-explicit-memory-representation)
        - [2.1.1 Free text](#211-free-text)
        - [2.1.2 Graph](#212-graph)
        - [2.1.3 Vector](#213-vector)
    - [2.2 Training with Explicit Memory](#22-training-with-explicit-memory)
        - [2.2.1 Pre-Training](#221-pre-training)
        - [2.2.2 Fine-Tuning](#222-fine-tuning)
    - [2.3 Training with externalized parameteric knowledge](#23-training-with-externalized-parameteric-knowledge)
        - [2.3.1 Long Contexts](#231-long-contexts)
        - [2.3.2 Knowledge Injection](#232-knowledge-injection)
    - [2.4 Limitations, Open Questions, Discussion](#24-limitations-open-questions-discussion)
- [3 Agentic Memory: Consolidating Memories into Humanic Agents](#3-agentic-memory-consolidating-memories-into-humanic-agents)
    - [3.1 Single-agent Memory](#31-single-agent-memory)
        - [3.1.1 Short-term Memory](#311-short-term-memory)
        - [3.1.2 Long-term Memory](#312-long-term-memory)
    - [3.2 Multi-agent Memory](#32-multi-agent-memory)
    - [3.3 System Architecture](#33-system-architecture)
        - [3.3.1 Data Ingestion](#331-data-ingestion)
        - [3.3.2 Storage and Retrieval](#332-storage-and-retrieval)
        - [3.3.3 User Interfaces and Application Invocation](#333-user-interfaces-and-application-invocation)
    - [3.4 Evaluation on Agent Memory](#34-evaluation-on-agent-memory)
        - [3.4.1 Qualitative Evaluation](#341-qualitative-evaluation)
        - [3.4.2 Quantitative Evaluation](#342-quantitative-evaluation)
    - [3.5 Limitations, Open Questions, Discussion](#35-limitations-open-questions-discussion)
- [4 Memory-augmented Multi-Modal Large Language models](#4-memory-augmented-multi-modal-large-language-models)
    - [4.1 Multimodal Context Modeling with Memory](#41-multimodal-context-modeling-with-memory)
        - [4.1.1 Audio Context Modeling](#411-audio-context-modeling)
        - [4.1.2 Video Context Modeling](#412-video-context-modeling)
        - [4.1.3 Other Modalities](#413-other-modalities)
    - [4.2 Downstream tasks](#42-downstream-tasks)
    - [4.3 Multimodal Contextual Memory for Robotics](#43-multimodal-contextual-memory-for-robotics)
        - [4.3.1 Multimodal Memory-Augmented Agents](#431-multimodal-memory-augmented-agents)
        - [4.3.2 Memory-Enhanced Navigation, Odometry, and Manipulation](#432-memory-enhanced-navigation-odometry-and-manipulation)
        - [4.3.3 Application](#433-application)
    - [4.4 Limitations, Open Questions, Discussion](#44-limitations-open-questions-discussion)




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
#### 2.1.1 Free text
- [PRCA: Fitting Black-Box Large Language Models for Retrieval Question Answering via Pluggable Reward-Driven Contextual Adapter](https://aclanthology.org/2023.emnlp-main.326/)
- [PaperQA: Retrieval-Augmented Generative Agent for Scientific Research](https://arxiv.org/abs/2312.07559)
- [Memory-assisted prompt editing to improve GPT-3 after deployment](https://arxiv.org/abs/2201.06009)
- [Prompt-Guided Retrieval Augmentation for Non-Knowledge-Intensive Tasks](https://aclanthology.org/2023.findings-acl.693/)
- [Enhancing LLM Intelligence with ARM-RAG: Auxiliary Rationale Memory for Retrieval Augmented Generation](https://arxiv.org/abs/2311.04177)
- [Understanding Retrieval Augmentation for Long-Form Question Answering](https://arxiv.org/abs/2310.12150)
- [Chain-of-Note: Enhancing Robustness in Retrieval-Augmented Language Models](https://aclanthology.org/2024.emnlp-main.813/)
- [IAG: Induction-Augmented Generation Framework for Answering Reasoning Questions](https://aclanthology.org/2023.emnlp-main.1/)
- [Retrieval meets Long Context Large Language Models](https://arxiv.org/abs/2310.03025)
- [RA-DIT: Retrieval-Augmented Dual Instruction Tuning](https://arxiv.org/abs/2310.01352)
- [RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval](https://arxiv.org/abs/2401.18059)
#### 2.1.2 Graph
- [ISEEQ: Information Seeking Question Generation using Dynamic Meta-Information Retrieval and Knowledge Graphs](https://arxiv.org/abs/2112.07622)
- [G-retriever: retrieval-augmented generation for textual graph understanding and question answering](https://dl.acm.org/doi/10.5555/3737916.3742140)
- [KnowledGPT: Enhancing Large Language Models with Retrieval and Storage Access on Knowledge Bases](https://arxiv.org/abs/2308.11761)
- [FABULA: Intelligence Report Generation Using Retrieval-Augmented Narrative Construction](https://dl.acm.org/doi/10.1145/3625007.3627505)
- [HyKGE: A Hypothesis Knowledge Graph Enhanced Framework for Accurate and Reliable Medical LLMs Responses](https://arxiv.org/abs/2312.15883)
- [Knowledge-Augmented Language Model Verification](https://arxiv.org/abs/2310.12836)
- [PIP-KAG: Mitigating Knowledge Conflicts in Knowledge-Augmented Generation via Parametric Pruning](https://arxiv.org/abs/2502.15543v1)
- [From RAG to Memory: Non-Parametric Continual Learning for Large Language Models](https://arxiv.org/abs/2502.14802)
- [Reducing Hallucinations in Language Model-based SPARQL Query Generation Using Post-Generation Memory Retrieval](https://arxiv.org/abs/2502.13369)
- [Reasoning on Graphs: Faithful and Interpretable Large Language Model Reasoning](https://arxiv.org/abs/2310.01061)
- [ProAI: Proactive Multi-Agent Conversational AI with Structured Knowledge Base for Psychiatric Diagnosis](https://arxiv.org/abs/2502.20689v1)
#### 2.1.3 Vector
- [LlamaIndex](https://github.com/jerryjliu/llama_index)
- [LangChain](https://github.com/langchain-ai/langchain)
- [Large Language Models Prompting With Episodic Memory](https://arxiv.org/abs/2408.07465)
- [Enhancing LLM Intelligence with ARM-RAG: Auxiliary Rationale Memory for Retrieval Augmented Generation](https://arxiv.org/abs/2311.04177)
- [Retrieve-and-Sample: Document-level Event Argument Extraction via Hybrid Retrieval Augmentation](https://aclanthology.org/2023.acl-long.17/)
- [Evaluating Very Long-Term Conversational Memory of LLM Agents](https://arxiv.org/abs/2402.17753)
### 2.2 Training with Explicit Memory
#### 2.2.1 Retrieval-based pretraining
- [REALM: Retrieval-Augmented Language Model Pre-Training](https://arxiv.org/abs/2002.08909)
- [Improving language models by retrieving from trillions of tokens](https://arxiv.org/abs/2112.04426)
- [Shall We Pretrain Autoregressive Language Models with Retrieval? A Comprehensive Study](https://aclanthology.org/2023.emnlp-main.482/)
- [InstructRetro: Instruction Tuning post Retrieval-Augmented Pretraining](https://arxiv.org/abs/2310.07713)
- [{$Memory^3$}: Language Modeling with Explicit Memory](https://arxiv.org/abs/2407.01178)
- [Atlas: Few-shot Learning with Retrieval Augmented Language Models](https://arxiv.org/abs/2208.03299)
- [Structure-Aware Language Model Pretraining Improves Dense Retrieval on Structured Data](https://arxiv.org/abs/2305.19912)
- [Copy Is All You Need](https://arxiv.org/abs/2307.06962)
#### 2.2.2 Retrieval-based fintuning
- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401)
- [Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection](https://arxiv.org/abs/2310.11511)
- [Leveraging Passage Retrieval with Generative Models for Open Domain Question Answering](https://aclanthology.org/2021.eacl-main.74/)
- [REPLUG: Retrieval-Augmented Black-Box Language Models](https://aclanthology.org/2024.naacl-long.463/)
- [Knowledge Graph-Augmented Language Models for Knowledge-Grounded Dialogue Generation](https://arxiv.org/abs/2305.18846)
- [UPRISE: Universal Prompt Retrieval for Improving Zero-Shot Evaluation](https://aclanthology.org/2023.emnlp-main.758/)
- [Self-Memory Alignment: Mitigating Factual Hallucinations with Generalized Improvement](https://arxiv.org/abs/2502.19127v1)
- [Interact, Instruct to Improve: A LLM-Driven Parallel Actor-Reasoner Framework for Enhancing Autonomous Vehicle Interactions](https://arxiv.org/abs/2503.00502)
- [Enhancing Retrieval-Augmented Large Language Models with Iterative Retrieval-Generation Synergy](https://aclanthology.org/2023.findings-emnlp.620/)
### 2.3 Training with externalized parameteric knowledge
#### 2.3.1 Long Contexts
- [Memorizing Transformers](https://arxiv.org/abs/2203.08913)
- [Unlimiformer: Long-Range Transformers with Unlimited Length Input](https://arxiv.org/abs/2305.01625)
- [Focused Transformer: Contrastive Training for Context Scaling](https://arxiv.org/abs/2307.03170)
- [EpMAN: Episodic Memory AttentioN for Generalizing to Longer Contexts](https://aclanthology.org/2025.acl-long.574/)
- [{$R^3$}Mem: Bridging Memory Retention and Retrieval via Reversible Compression](https://aclanthology.org/2025.findings-acl.235/)
- [Augmenting Language Models with Long-Term Memory](https://arxiv.org/abs/2306.07174)
#### 2.3.2 Knowledge Injection
- [Training language models with memory augmentation](https://arxiv.org/abs/2205.12674)
- [Lift Yourself Up: Retrieval-augmented Text Generation with Self Memory](https://arxiv.org/abs/2305.02437)
- [Faithful reasoning using large language models](https://arxiv.org/abs/2208.14271)
### 2.4 Limitations, Open Questions, Discussion
- [Efficient Multivariate Time Series Forecasting via Calibrated Language Models with Privileged Knowledge Distillation](https://arxiv.org/abs/2505.02138)
- [Towards Cross-Modality Modeling for Time Series Analytics: A Survey in the LLM Era](https://arxiv.org/abs/2505.02583)
- [Foundation Models for Spatio-Temporal Data Science: A Tutorial and Survey](https://arxiv.org/abs/2503.13502)







## 3 Agentic Memory: Consolidating Memories into Humanic Agents

- [Human memory: A proposed system and its control processes](https://app.nova.edu/toolbox/instructionalproducts/edd8124/articles/1968-Atkinson_and_Shiffrin.pdf) 
- [A survey on the memory mechanism of large language model based agents](https://arxiv.org/abs/2404.13501) 

### 3.1 Single-agent Memory
- [LLM as a robotic brain: Unifying egocentric memory and control](https://arxiv.org/abs/2304.09349) 
- [Evaluating very long-term conversational memory of LLM agents](https://arxiv.org/abs/2402.17753) 
- [Memory3: Language modeling with explicit memory](https://arxiv.org/abs/2407.01178) 



#### 3.1.1 Short-term Memory
- [Chain-of-thought prompting elicits reasoning in large language models](https://arxiv.org/abs/2201.11903) 
- [Self-consistency improves chain of thought reasoning in language models](https://arxiv.org/abs/2203.11171)
- [Tree of thoughts: Deliberate problem solving with large language models](https://arxiv.org/abs/2305.10601) 
- [Graph of thoughts: Solving elaborate problems with large language models](https://arxiv.org/abs/2308.09687) 
- [ReAct: Synergizing reasoning and acting in language models](https://arxiv.org/abs/2210.03629) 
- [Reflexion: Language agents with verbal reinforcement learning](https://arxiv.org/abs/2303.11366) 
- [Reflectevo: Improving meta introspection of small LLMs by learning self-reflection](https://arxiv.org/abs/2505.16475) 
- [Mars: Situated inductive reasoning in an open-world environment](https://arxiv.org/abs/2410.08126) 
- [RAM: Towards an ever-improving memory system by learning from communications](https://arxiv.org/abs/2404.12045) 
- [Absolute zero: Reinforced self-play reasoning with zero data](https://arxiv.org/abs/2505.03335) 
- [Metareflection: Learning instructions for language agents using past reflections](https://arxiv.org/abs/2405.13009) 
- [Learn beyond the answer: Training language models with reflection for mathematical reasoning](https://arxiv.org/abs/2406.12050) 
- [RoT: Enhancing large language models with reflection on search trees](https://arxiv.org/abs/2404.05449) 
- [Mirror: A multiple-perspective self-reflection method for knowledge-rich reasoning](https://arxiv.org/abs/2402.14963) 
- [Textgrad: Automatic differentiation via text](https://arxiv.org/abs/2406.07496) 

#### 3.1.2 Long-term Memory
- [Langsuite: Planning, controlling and interacting with large language models in embodied text environments](https://arxiv.org/abs/2406.16294) 
- [Benchmarking the spectrum of agent capabilities](https://arxiv.org/abs/2109.06780) 
- [The arcade learning environment: An evaluation platform for general agents](https://arxiv.org/abs/1207.4708) 
- [Understanding the planning of LLM agents: A survey](https://arxiv.org/abs/2402.02716) 
- [Generative agents: Interactive simulacra of human behavior](https://arxiv.org/abs/2304.03442) 
- [Recmind: Large language model powered agent for recommendation](https://arxiv.org/abs/2308.14296) 
- [Large language models are semi-parametric reinforcement learning agents](https://arxiv.org/abs/2403.01021) 
- [How do large language models capture the ever-changing world knowledge? a review of recent advances](https://arxiv.org/abs/2310.07343) 
- [From static to dynamic: A continual learning framework for large language models](https://arxiv.org/abs/2310.14248) 
- [Ret-LLM: Towards a general read-write memory for large language models](https://arxiv.org/abs/2305.14322) 
- [Empowering working memory for large language model agents](https://arxiv.org/abs/2312.17259) 
- [Think-in-memory: Recalling and post-thinking enable LLMs with long-term memory](https://arxiv.org/abs/2311.08719) 
- [RAP: Retrieval-augmented planning with contextual memory for multimodal LLM agents](https://arxiv.org/abs/2402.03610) 
- [Learn-by-interact: A data-centric framework for self-adaptive agents in realistic environments](https://arxiv.org/abs/2501.10893) 
- [Memp: Exploring agent procedural memory](https://arxiv.org/abs/2508.06433) 
- [Contextual experience replay for self-improvement of language agents](https://arxiv.org/abs/2411.02534) 
- [Sage: Self-evolving agents with reflective and memory-augmented abilities](https://arxiv.org/abs/2501.12345) 
- [Reasoningbank: Scaling agent self-evolving with reasoning memory](https://arxiv.org/abs/2509.25140) 
- [Memory-assisted prompt editing to improve GPT-3 after deployment](https://arxiv.org/abs/2201.06009) 
- [Learning to repair: Repairing model output errors after deployment using a dynamic memory of feedback](https://arxiv.org/abs/2112.09737) 
- [Towards teachable reasoning systems: Using a dynamic memory of user feedback for continual system improvement](https://arxiv.org/abs/2204.13074) 
- [Compress to impress: Unleashing the potential of compressive memory in real-world long-term conversations](https://arxiv.org/abs/2402.11975) 
- [KnowledGPT: Enhancing large language models with retrieval and storage access on knowledge bases](https://arxiv.org/abs/2308.11761) 
- [Memory-augmented LLM personalization with short-and long-term memory coordination](https://arxiv.org/abs/2309.11696) 
- [MemoryBank: Enhancing large language models with long-term memory](https://arxiv.org/abs/2305.10250) 




### 3.2 Multi-agent Memory
- [A-mem: Agentic memory for llm agents](https://arxiv.org/abs/2502.12110)
- [Internet of agents: Weaving a web of heterogeneous agents for collaborative intelligence](https://arxiv.org/abs/2407.07061)
- [LLM-Powered Decentralized Generative Agents with Adaptive Hierarchical Knowledge Graph for Cooperative Planning](https://arxiv.org/abs/2502.05453)
- [Memory Sharing for Large Language Model based Agents](https://arxiv.org/abs/2404.09982)
- [Zep: A Temporal Knowledge Graph Architecture for Agent Memory](https://arxiv.org/abs/2501.13956)



### 3.3 System Architecture
- [LangChain](https://github.com/langchain-ai/langchain) 
- [LlamaIndex](https://github.com/run-llama/llama_index) 
- [Zep: A temporal knowledge graph architecture for agent memory](https://arxiv.org/abs/2501.13956) 
- [Sentence-BERT: Sentence embeddings using siamese BERT-networks](https://arxiv.org/abs/1908.10084) 




### 3.4 Evaluation on Agent Memory
- [The NarrativeQA Reading Comprehension Challenge](https://arxiv.org/abs/1712.07040) 
- [Quality: Question answering with long input texts, yes!](https://arxiv.org/abs/2112.08608) 
- [LooGle: Can long-context language models understand long contexts?](https://arxiv.org/abs/2311.04939) 
- [RetrievalQA: Assessing adaptive retrieval-augmented generation for short-form open-domain question answering](https://arxiv.org/abs/2402.16457) 
- [Needle In A Haystack](https://github.com/gkamradt/LLMTest_NeedleInAHaystack) 
- [ALFWorld: Aligning text and embodied environments for interactive learning](https://arxiv.org/abs/2010.03768) 
- [WebArena: A realistic web environment for building autonomous agents](https://arxiv.org/abs/2307.13854) 
- [Gentopia: A collaborative platform for tool-augmented LLMs](https://arxiv.org/abs/2308.04030) 
- [AgentBench: Evaluating LLMs as agents](https://arxiv.org/abs/2308.03688) 
- [LongMemEval: Benchmarking chat assistants on long-term interactive memory](https://arxiv.org/abs/2410.10813) 
- [The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783) 
- [Generative agent-based modeling with actions grounded in physical, social, or digital space using Concordia](https://arxiv.org/abs/2312.03664) 

### 3.5 Limitations, Open Questions, Discussion
- [A Scalable Communication Protocol for Networks of Large Language Models](https://arxiv.org/abs/2410.11905)
- [A survey on the memory mechanism of large language model based agents](https://arxiv.org/abs/2404.13501)
- [DroidSpeak: Enhancing Cross-LLM Communication](https://arxiv.org/abs/2411.02820)
- [IBGP: Imperfect Byzantine Generals Problem for Zero-Shot Robustness in Communicative Multi-Agent Systems](https://arxiv.org/abs/2410.16237)
- [RAM: Towards an Ever-Improving Memory System by Learning from Communications](https://arxiv.org/abs/2404.12045)
- [Rethinking memory in ai: Taxonomy, operations, topics, and future directions](https://arxiv.org/abs/2505.00675)

## 4 Memory-augmented Multi-Modal Large Language models

### 4.1 Multimodal Context Modeling with Memory
#### 4.1.1 Audio Context Modeling
- [Memory-augmented conformer for improved end-to-end long-form ASR](https://dblp.org/rec/conf/interspeech/CarvalhoA23.html)
- [Loop Copilot: Conducting AI Ensembles for Music Generation and Iterative Editing](https://dblp.org/rec/journals/corr/abs-2310-12404.html)
- [MR-MT3: Memory Retaining Multi-Track Music Transcription to Mitigate Instrument Leakage](https://dblp.org/rec/journals/corr/abs-2403-10024.html)



#### 4.1.2 Video Context Modeling
##### Large Video Language Models
- [Memory Consolidation Enables Long-Context Video Understanding](https://dblp.org/rec/journals/corr/abs-2402-05861.html)
- [MovieChat: From Dense Token to Sparse Memory for Long Video Understanding](https://dblp.org/rec/journals/corr/abs-2307-16449.html)
- [MA-LMM: Memory-Augmented Large Multimodal Model for Long-Term Video Understanding](https://dblp.org/rec/journals/corr/abs-2404-05726.html)
- [Streaming Long Video Understanding with Large Language Models](https://dblp.org/rec/journals/corr/abs-2405-16009.html)
- [Flash-VStream: Memory-Based Real-Time Understanding for Long Video Streams](https://dblp.org/rec/journals/corr/abs-2406-08085.html)
- [Videollamb: Long-context video understanding with recurrent memory bridges](https://arxiv.org/abs/2409.01071)
- [OmniDrive: A Holistic LLM-Agent Framework for Autonomous Driving with 3D Perception, Reasoning and Planning](https://arxiv.org/abs/2405.01533)

##### Video Agents
- [A Simple LLM Framework for Long-Range Video Question-Answering](https://dblp.org/rec/journals/corr/abs-2312-17235.html)
- [LifelongMemory: Leveraging LLMs for Answering Queries in Egocentric Videos](https://dblp.org/rec/journals/corr/abs-2312-05269.html)
- [VideoAgent: Long-form Video Understanding with Large Language Model as Agent](https://dblp.org/rec/journals/corr/abs-2403-10517.html)
- [VideoAgent: A Memory-augmented Multimodal Agent for Video Understanding](https://dblp.org/rec/journals/corr/abs-2403-11481.html)
- [ChatVideo: A Tracklet-centric Multimodal and Versatile Video Understanding System](https://dblp.org/rec/journals/corr/abs-2304-14407.html)
- [DoraemonGPT: Toward Understanding Dynamic Scenes with Large Language Models](https://dblp.org/rec/journals/corr/abs-2401-08392.html)
- [Language Repository for Long Video Understanding](https://dblp.org/rec/journals/corr/abs-2403-14622.html)
- [DrVideo: Document Retrieval Based Long Video Understanding](https://dblp.org/rec/journals/corr/abs-2406-12846.html)


#### 4.1.3 Other Modalities
- [Attend to You: Personalized Image Captioning with Context Sequence Memory Networks](https://dblp.org/rec/conf/cvpr/ParkKK17.html)
- [Divide-and-Assemble: Learning Block-wise Memory for Unsupervised Anomaly Detection](https://dblp.org/rec/conf/iccv/HouZZXPZ21.html)
- [Memory-Guided Unsupervised Image-to-Image Translation](https://dblp.org/rec/conf/cvpr/JeongKLS21.html)
- [Coloring With Limited Data: Few-Shot Colorization via Memory Augmented Networks](https://dblp.org/rec/conf/cvpr/YooBCLCC19.html)
- [Universal Face Restoration With Memorized Modulation](https://dblp.org/rec/journals/corr/abs-2110-01033.html)
- [Learning Meta-class Memory for Few-Shot Semantic Segmentation](https://dblp.org/rec/conf/iccv/WuSL021.html)
- [Remember the Difference: Cross-Domain Few-Shot Semantic Segmentation via Meta-Memory Transfer](https://dblp.org/rec/conf/cvpr/WangDWEFZ22.html)
- [Memory Oriented Transfer Learning for Semi-Supervised Image Deraining](https://dblp.org/rec/conf/cvpr/HuangYH21.html)
- [TransformerFusion: Monocular RGB Scene Reconstruction using Transformers](https://dblp.org/rec/conf/nips/BozicPTDN21.html)

### 4.2 Downstream tasks
##### Object-level
- [Learning Video Object Segmentation with Visual Memory](https://doi.org/10.1109/ICCV.2017.480)
- [Recurrent Filter Learning for Visual Tracking](https://dblp.org/rec/conf/iccvw/YangC17.html)
- [Learning Recurrent Memory Activation Networks for Visual Tracking](https://dblp.org/rec/journals/tip/PuSMZY21.html)
- [Learning Dynamic Memory Networks for Object Tracking](https://dblp.org/rec/conf/eccv/YangC18.html)
- [Video Object Segmentation Using Space-Time Memory Networks](https://dblp.org/rec/conf/iccv/OhLXK19.html)
- [Rethinking Space-Time Networks with Improved Memory Coverage for Efficient Video Object Segmentation](https://dblp.org/rec/conf/nips/ChengTT21.html)
- [STMTrack: Template-Free Visual Tracking With Space-Time Memory Networks](https://dblp.org/rec/conf/cvpr/FuLFW21.html)
- [Video Object Segmentation with Episodic Graph Memory Networks](https://dblp.org/rec/conf/eccv/LuWDZSG20.html)
- [Dual Temporal Memory Network for Efficient Video Object Segmentation](https://dblp.org/rec/conf/mm/ZhangWLL0L20.html)
- [Alignment Before Aggregation: Trajectory Memory Retrieval Network for Video Object Segmentation](https://dblp.org/rec/conf/iccv/SunWMZW23.html)
- [Associating Objects with Transformers for Video Object Segmentation](https://dblp.org/rec/conf/nips/YangWY21.html)
- [Local Memory Attention for Fast Video Semantic Segmentation](https://dblp.org/rec/conf/iros/PaulDGT21.html)
- [TrackFormer: Multi-Object Tracking with Transformers](https://dblp.org/rec/conf/cvpr/MeinhardtKLF22.html)
- [XMem: Long-Term Video Object Segmentation with an Atkinson-Shiffrin Memory Model](https://dblp.org/rec/conf/eccv/ChengS22.html)
- [Streaming Video Model](https://dblp.org/rec/conf/cvpr/ZhaoLTCCZ23.html)
- [Multiscale Memory Comparator Transformer for Few-Shot Video Segmentation](https://dblp.org/rec/journals/corr/abs-2307-07812.html)
- [Reading Relevant Feature from Global Representation Memory for Visual Object Tracking](https://dblp.org/rec/conf/nips/ZhouGHL0GZ23.html)
- [RMem: Restricted Memory Banks Improve Video Object Segmentation](https://dblp.org/rec/journals/corr/abs-2406-08476.html)
- [Efficient Video Object Segmentation via Modulated Cross-Attention Memory](https://dblp.org/rec/journals/corr/abs-2403-17937.html)
- [SAM 2: Segment Anything in Images and Videos](https://arxiv.org/abs/2408.00714)

##### Action-level
- [Long Short-Term Transformer for Online Action Detection](https://dblp.org/rec/conf/nips/XuXCLXTS21.html)
- [MeMViT: Memory-Augmented Multiscale Vision Transformer for Efficient Long-Term Video Recognition](https://dblp.org/rec/conf/cvpr/WuLM0XMF22.html)
- [Recurring the Transformer for Video Action Recognition](https://dblp.org/rec/conf/cvpr/YangDLZSY22.html)
- [Memory-and-Anticipation Transformer for Online Action Understanding](https://dblp.org/rec/conf/iccv/WangCHWL23.html)
- [Online Temporal Action Localization with Memory-Augmented Transformer](https://arxiv.org/abs/2408.02957)

##### General-level
- [Video Summarization with Long Short-Term Memory](https://dblp.org/rec/conf/eccv/ZhangCSG16.html)
- [Extractive Video Summarizer with Memory Augmented Neural Networks](https://dblp.org/rec/conf/mm/FengLKZ18.html)
- [Stacked Memory Network for Video Summarization](https://dblp.org/rec/conf/mm/Wang000FT19.html)
- [M3: Multimodal Memory Modelling for Video Captioning](https://dblp.org/rec/conf/cvpr/Wang000T18.html)
- [Memory-Attended Recurrent Network for Video Captioning](https://dblp.org/rec/conf/cvpr/PeiZWKST19.html)
- [MART: Memory-Augmented Recurrent Transformer for Coherent Video Paragraph Captioning](https://dblp.org/rec/conf/acl/LeiWSYBB20.html)
- [AAP-MIT: Attentive Atrous Pyramid Network and Memory Incorporated Transformer for Multisentence Video Description](https://dblp.org/rec/journals/tip/PrudvirajRVM22.html)
- [Heterogeneous Memory Enhanced Multimodal Attention Model for Video Question Answering](https://dblp.org/rec/conf/cvpr/FanZZW0H19.html)
- [A Read-Write Memory Network for Movie Story Understanding](https://dblp.org/rec/conf/iccv/NaLKK17.html)
- [Glance and Focus: Memory Prompting for Multi-Event Video Question Answering](https://dblp.org/rec/conf/nips/Bai0023.html)

##### Others 
- [MemFlow: Optical Flow Estimation and Prediction with Memory](https://dblp.org/rec/journals/corr/abs-2404-04808.html)
- [MAMo: Leveraging Memory and Attention for Monocular Video Depth Estimation](https://dblp.org/rec/conf/iccv/YasarlaCJSGP23.html)
- [Multi-Scale Memory-Based Video Deblurring](https://dblp.org/rec/conf/cvpr/JiY22.html)
- [MENet: A Memory-Based Network with Dual-Branch for Efficient Event Stream Processing](https://dblp.org/rec/conf/eccv/SunZCCL22.html)
- [CroMM-VSR: Cross-Modal Memory Augmented Visual Speech Recognition](https://dblp.org/rec/journals/tmm/KimHPR22.html)
- [Distinguishing Homophenes Using Multi-Head Visual-Audio Memory for Lip Reading](https://dblp.org/rec/conf/aaai/KimYR22.html)

##### Generation
- [Video Prediction Recalling Long-Term Motion Context via Memory Alignment Learning](https://dblp.org/rec/conf/cvpr/LeeKCKR21.html)
- [MV-TON: Memory-based Video Virtual Try-on network](https://dblp.org/rec/conf/mm/ZhongWTLW21.html)
- [Audio-driven Talking Face Video Generation with Natural Head Pose](https://arxiv.org/abs/2002.10137)
- [SyncTalkFace: Talking Face Generation with Precise Lip-Syncing via Audio-Lip Memory](https://dblp.org/rec/conf/aaai/ParkKHCR22.html)
- [EMMN: Emotional Motion Memory Network for Audio-driven Emotional Talking Face Generation](https://dblp.org/rec/conf/iccv/TanJP23.html)
- [STAM: A SpatioTemporal Attention Based Memory for Video Prediction](https://dblp.org/rec/journals/tmm/00020WM023.html)
- [Memories are One-to-Many Mapping Alleviators in Talking Face Generation](https://dblp.org/rec/journals/corr/abs-2212-05005.html)

  
### 4.3 Multimodal Contextual Memory for Robotics
#### 4.3.1 Multimodal Memory-Augmented Agents
- [JARVIS-1: Open-World Multi-task Agents with Memory-Augmented Multimodal Language Models](https://dblp.org/rec/journals/corr/abs-2311-05997.html)
- [GPT-4V in Wonderland: Large Multimodal Models for Zero-Shot Smartphone GUI Navigation](https://dblp.org/rec/journals/corr/abs-2311-07562.html)
- [Explore, Select, Derive, and Recall: Augmenting LLM with Human-like Memory for Mobile Task Automation](https://dblp.org/rec/journals/corr/abs-2312-03003.html)
- [AppAgent: Multimodal Agents as Smartphone Users](https://dblp.org/rec/journals/corr/abs-2312-13771.html)
- [OS-Copilot: Towards Generalist Computer Agents with Self-Improvement](https://dblp.org/rec/journals/corr/abs-2402-07456.html)
- [Multimodal Embodied Interactive Agent for Cafe Scene](https://dblp.org/rec/journals/corr/abs-2402-00290.html)

#### 4.3.2 Memory-Enhanced Navigation, Odometry, and Manipulation
- [Visual Memory for Robust Path Following](https://dblp.org/rec/conf/nips/KumarGFLM18.html)
- [Structured Scene Memory for Vision-Language Navigation](https://dblp.org/rec/conf/cvpr/WangWLXS21.html)
- [Visual Graph Memory with Unsupervised Representation for Visual Navigation](https://dblp.org/rec/conf/iccv/KwonKCYPO21.html)
- [MemoNav: Working Memory Model for Visual Navigation](https://dblp.org/rec/journals/corr/abs-2402-19161.html)
- [Scene Memory Transformer for Embodied Agents in Long-Horizon Tasks](https://dblp.org/rec/conf/cvpr/FangT0S19.html)
- [MultiON: Benchmarking Semantic Map Memory using Multi-Object Navigation](https://dblp.org/rec/conf/nips/WaniPJCS20.html)
- [Deep Visual Odometry With Adaptive Memory](https://dblp.org/rec/journals/pami/XueWWZ22.html)
- [Out of Sight, Still in Mind: Reasoning and Planning about Unobserved Objects with Video Tracking Enabled Memory Models](https://dblp.org/rec/conf/icra/HuangYKPCLH24.html)



