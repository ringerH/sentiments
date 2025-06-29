#!/bin/bash

# Create the complete project structure
echo "Creating MLOps project structure..."

# Create main directories
mkdir -p sentiment-mlops/{src/{data,models,utils,config},scripts,tests,docker,k8s,ci-cd/{.github/workflows,argo-workflows},monitoring/{prometheus,grafana}}

cd sentiment-mlops

# Create __init__.py files for Python packages
touch src/__init__.py
touch src/data/__init__.py
touch src/models/__init__.py
touch src/utils/__init__.py
touch src/config/__init__.py

# Create main script files
touch scripts/train.py
touch scripts/predict.py
touch scripts/evaluate.py

# Create test files
touch tests/test_preprocessor.py
touch tests/test_trainer.py
touch tests/__init__.py

# Create Docker files
touch docker/Dockerfile.train
touch docker/Dockerfile.serve
touch docker/requirements.txt

# Create Kubernetes manifests
touch k8s/training-job.yaml
touch k8s/serving-deployment.yaml
touch k8s/configmap.yaml

# Create CI/CD files
touch ci-cd/Jenkinsfile
touch ci-cd/.github/workflows/ci.yml
touch ci-cd/argo-workflows/training-pipeline.yaml

# Create monitoring configs
touch monitoring/prometheus/prometheus.yml
touch monitoring/grafana/dashboard.json

# Create main project files
touch README.md
touch requirements.txt

echo "Project structure created successfully!"
echo "Next: Extract classes from notebook into src/ modules"