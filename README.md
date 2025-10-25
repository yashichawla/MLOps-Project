# Break The Bot - MLOps Project

## 📌 Introduction

Large Language Models (LLMs) are increasingly deployed in real-world applications, but they remain vulnerable to jailbreaks and prompt-injection attacks.  
Our project, **Break The Bot**, aims to build an automated MLOps pipeline for continuous safety evaluation of LLMs.

### System will:

- Preprocess and run adversarial prompts
- Measure **Attack Success Rate (ASR)** and **Refusal Quality**
- Use **LLM-as-a-Judge** for automated scoring
- Store and visualize results on dashboards
- Integrate with **CI/CD pipelines** to block unsafe releases

### Team Members:

1. Anjali Pai
2. Atharv Talnikar
3. Nitya Ravi
4. Rahul Kulkarni
5. Taniksha Datar
6. Yashi Chawla

### Repository Structure

```plaintext
MLOps Project/
├── data/ # seed prompts
├── pipelines/ # evaluation and monitoring pipelines
├── runner/ # evaluator service (FastAPI, executes prompts on target LLMs)
├── judge/ # judge service (FastAPI, scoring unsafe completions)
├── dashboards/ # Grafana dashboards, metrics visualization
├── docs/ # documentation, reports, diagrams
├── tests/ # unit tests, smoke tests, regression suites
├── .github/
│ └── workflows/ # CI/CD automation (GitHub Actions)
│
├── README.md # project overview
├── requirements.txt# Python dependencies
├── .gitignore # ignored files (logs, envs, cache, etc.)
└── docker-compose.yml # container orchestration
```

### Setup Instructions:

1. Clone Repository

```bash
git clone https://github.com/yashichawla/MLOps-Project
cd MLOps-Project
```

2. Create Virtual Environment

```bash
   python -m venv venv
   source venv/bin/activate # On Mac/Linux
   venv\Scripts\activate # On Windows
```

3. Install Dependencies

```bash
   pip install -r requirements.txt
```

4. Run with Docker

```bash
   docker-compose up --build
```

### Features:

- Prompt Generator: Generates adversarial prompts using attacker LLMs.
- Evaluator Service: Runs prompts against target models and logs results.
- Judge Service: Scores responses for safety and refusal quality.
- Dashboards: Grafana visualizations for safety trends, ASR, and alerts.
- CI/CD Integration: Blocks unsafe deployments if safety metrics fail.
- Failure Analysis: Clustering and regression testing of jailbreak cases.

### Key Metrics:

- Attack Success Rate (ASR) - % of successful jailbreaks.
- Refusal Quality - judged clarity and robustness of refusals.
- Coverage Metrics - number and diversity of tested adversarial prompts.

### Project Timeline:

- Week 1-2: Repo setup, governance policy, seed prompt generation.
- Week 3-4: Prompt generator + evaluator API.
- Week 5-6: Judge API + calibration with human labels.
- Week 7-8: Dashboards, monitoring, failure analysis.
- Week 9-10: CI/CD gates, final validation, and reporting.
