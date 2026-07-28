import random
from typing import Tuple, List

# Define the categories in order
CATEGORIES = [
    "Artificial Intelligence",
    "Machine Learning",
    "Computer Vision",
    "Natural Language Processing",
    "Robotics",
    "Cyber Security",
    "Cloud Computing"
]

# Base vocab/templates for synthetic abstract generation
TEMPLATES = {
    "Artificial Intelligence": [
        "This paper proposes a new cognitive architecture for artificial general intelligence based on reasoning agents.",
        "We explore search algorithms and heuristic approaches to solve complex decision-making problems in AGI systems.",
        "An investigation into symbolic reasoning and knowledge representation in modern artificial intelligence systems.",
        "We discuss the history of the Turing test and outline next-generation agentic frameworks and reasoning models.",
        "A study on constraint satisfaction problems and path-finding heuristics in intelligent game-playing software.",
        "We introduce an autonomous agent architecture that models human-like cognitive processes and decision logic.",
        "Research on heuristic search, multi-agent systems, and planning models under high uncertainty environments.",
        "An overview of cognitive systems, reasoning paradigms, and knowledge graphs in semantic AI applications.",
        "This research designs an agentic system that simulates human problem-solving through modular reasoning structures.",
        "We present a survey of artificial intelligence frameworks focusing on logic programming, search space optimization, and expert systems."
    ],
    "Machine Learning": [
        "A novel approach to optimize deep neural networks using adaptive learning rate scheduling and gradient clipping.",
        "We analyze supervised learning algorithms and compare linear regression with support vector machines on high-dimensional data.",
        "This study addresses the problem of overfitting in deep networks by introducing a regularized loss function.",
        "We present hyperparameter tuning strategies for random forests and gradient boosted decision trees.",
        "An empirical evaluation of reinforcement learning algorithms in dynamic, continuous action spaces.",
        "This paper presents a model-agnostic technique for explaining predictions of complex black-box machine learning models.",
        "We investigate transfer learning models and fine-tuning configurations to improve accuracy under low data regimes.",
        "A deep neural network architecture designed for regression and multi-class classification tasks on structured tabular data.",
        "We formulate an optimization algorithm that speeds up gradient descent convergence in deep learning models.",
        "Research focusing on unsupervised learning, clustering algorithms, dimensionality reduction, and autoencoders."
    ],
    "Computer Vision": [
        "We present a real-time object detection framework using convolutional neural networks and bounding box regression.",
        "This paper introduces a new model for semantic image segmentation based on U-Net and pixel-level classification.",
        "An analysis of convolutional layers and feature map visualizations in deep neural networks for facial recognition.",
        "We develop a YOLO-based system for autonomous vehicle perception, detecting pedestrians, traffic lights, and vehicles.",
        "A novel approach to optical character recognition (OCR) in degraded historical documents using deep vision models.",
        "We study image super-resolution and generative adversarial networks for enhancing low-resolution medical images.",
        "Research in active contours, edge detection, feature extraction, and scale-invariant feature transform (SIFT).",
        "A convolutional network designed for video classification and action recognition using 3D convolutions.",
        "We introduce a technique for depth estimation and stereo vision matching using local descriptor learning.",
        "We present pixel-level anomaly detection in industrial manufacturing pipelines using convolutional autoencoders."
    ],
    "Natural Language Processing": [
        "A transformer-based language model for neural machine translation and sequence-to-sequence learning.",
        "We investigate text tokenization techniques and vocabulary size optimization in large language models.",
        "This paper describes a sentiment analysis pipeline that uses word embeddings and bidirectional LSTM networks.",
        "We present a method for named entity recognition (NER) and part-of-speech (POS) tagging in biomedical text.",
        "A study on query understanding, document retrieval, and semantic search models using pre-trained sentence transformers.",
        "We introduce a sequence-to-sequence model for abstractive text summarization and automated headline generation.",
        "Research on language modeling, syntax parsing, dependency grammar, and semantic parsing of natural language queries.",
        "An evaluation of contextualized word representation models like BERT on question-answering benchmarks.",
        "We present a framework for fine-tuning pre-trained generative text models with low-rank adaptation.",
        "A natural language processing pipeline for identifying topics in unstructured document collections using topic modeling."
    ],
    "Robotics": [
        "We study inverse kinematics and trajectory planning for a six-degree-of-freedom robotic manipulator arm.",
        "This paper presents a sensor fusion approach combining LiDAR and camera data for robot localization and SLAM.",
        "We develop a path planning algorithm for mobile robots operating in cluttered, dynamic environments.",
        "An investigation into autonomous drone navigation, obstacle avoidance, and PID flight control loops.",
        "We explore human-robot interaction, force feedback control, and haptic teleoperation systems.",
        "A reinforcement learning approach to robot locomotion, gait control, and balancing on rough terrain.",
        "Research on actuators, feedback loops, odometry sensors, and wheel encoders in robotic platforms.",
        "We propose a decentralized control protocol for robotic swarms performing search and rescue operations.",
        "An implementation of ROS (Robot Operating System) nodes for navigation, mapping, and sensor integration.",
        "A study on grasping algorithms and tactile sensor arrays for robotic hands manipulating fragile objects."
    ],
    "Cyber Security": [
        "We analyze cryptographic protocols and secure key exchange mechanisms for internet of things devices.",
        "This paper presents a machine learning system for network intrusion detection and malware classification.",
        "We identify common software vulnerabilities like SQL injection, cross-site scripting (XSS), and buffer overflows.",
        "An investigation into phishing attack prevention, spam filtering, and email authentication standards.",
        "We design a secure zero-trust architecture for enterprise networks, implementing multi-factor authentication.",
        "A study on malware sandbox analysis, dynamic binary analysis, and reverse engineering techniques.",
        "We explore firewall configurations, intrusion prevention systems, and virtual private network (VPN) protocols.",
        "Research on ransomware detection mechanisms, file system monitoring, and threat intelligence sharing.",
        "A penetration testing framework for discovering security loopholes in web applications and API endpoints.",
        "We analyze the security of blockchain consensus protocols and smart contract vulnerabilities."
    ],
    "Cloud Computing": [
        "We propose an auto-scaling algorithm for cloud virtual machines to optimize resource utilization and billing costs.",
        "This paper describes a microservices architecture deployed on Kubernetes with dynamic load balancing.",
        "We analyze serverless execution models (Function-as-a-Service) and cold start latency in cloud platforms.",
        "An investigation into cloud storage bucket security, access control policies, and data encryption at rest.",
        "We explore containerization technologies like Docker and compare container engines on production workloads.",
        "A multi-cloud deployment strategy utilizing AWS, GCP, and Azure for high availability and disaster recovery.",
        "Research on distributed databases, virtual networks, software-defined networking, and cloud storage architectures.",
        "We present a monitoring framework for tracking resource usage, memory, and CPU in large cloud clusters.",
        "An implementation of infrastructural automation (Infrastructure as Code) using Terraform and Ansible.",
        "We evaluate database replication strategies, network latency, and high-performance computing in cloud environments."
    ]
}

# Subphrase components to mix and match synthetically to expand data size
PHRASES = {
    "Artificial Intelligence": ["agentic system", "heuristic search", "symbolic logic", "AGI", "reasoning model", "cognitive agent", "decision tree reasoning", "knowledge graph"],
    "Machine Learning": ["gradient descent", "neural net", "deep learning model", "supervised classification", "random forest", "hyperparameter optimizer", "cross validation", "loss minimization"],
    "Computer Vision": ["bounding boxes", "pixel classification", "image features", "YOLO object detection", "CNN model", "semantic segmentation", "optical flow", "visual tracking"],
    "Natural Language Processing": ["text translation", "sentence transformer", "token vocabulary", "word embeddings", "LLM", "named entities", "sentiment analyzer", "attention mechanism"],
    "Robotics": ["manipulator arm", "SLAM mapping", "LiDAR navigation", "sensor fusion", "kinematic model", "ROS node", "PID controller", "actuator feedback"],
    "Cyber Security": ["phishing email", "malware execution", "SQL injection", "XSS attack", "firewall rules", "cryptographic keys", "vulnerability scan", "intrusion alert"],
    "Cloud Computing": ["docker container", "kubernetes pod", "autoscaling instance", "cloud bucket storage", "serverless lambda function", "virtual machine networking", "microservice api", "terraform deployment"]
}

def generate_synthetic_dataset(samples_per_category: int = 150) -> Tuple[List[str], List[int], List[str], List[int]]:
    """
    Generates a dataset of synthetic technical documents abstracts for training and validation.
    """
    train_texts = []
    train_labels = []
    val_texts = []
    val_labels = []

    for cat_idx, category in enumerate(CATEGORIES):
        templates = TEMPLATES[category]
        phrases = PHRASES[category]
        
        # Generate samples
        samples = []
        for _ in range(samples_per_category):
            # Pick a base template
            base = random.choice(templates)
            # Pick a couple of target phrases to inject or append
            extra_phrases = random.sample(phrases, 2)
            # Generate a varied paragraph
            paragraph = f"{base} In this work, we leverage {extra_phrases[0]} and optimize {extra_phrases[1]} to establish state-of-the-art results in the domain of {category.lower()}."
            samples.append(paragraph)
        
        # Shuffle and split 80/20
        random.shuffle(samples)
        split_idx = int(samples_per_category * 0.8)
        
        for text in samples[:split_idx]:
            train_texts.append(text)
            train_labels.append(cat_idx)
            
        for text in samples[split_idx:]:
            val_texts.append(text)
            val_labels.append(cat_idx)

    # Shuffle the final lists synchronously
    train_combined = list(zip(train_texts, train_labels))
    random.shuffle(train_combined)
    train_texts, train_labels = zip(*train_combined)

    val_combined = list(zip(val_texts, val_labels))
    random.shuffle(val_combined)
    val_texts, val_labels = zip(*val_combined)

    return list(train_texts), list(train_labels), list(val_texts), list(val_labels)
