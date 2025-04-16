#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status

# Configuration
PROJECT_ID=$(gcloud config get-value project)
CLUSTER_NAME="my-api-cluster"
REGION="us-central1-a"
MACHINE_TYPE="e2-standard-2"
NODE_COUNT=4

echo "==== Starting deployment process to Google Cloud ===="
echo "Project ID: $PROJECT_ID"
echo "Cluster: $CLUSTER_NAME in $REGION"
echo ""

# Step 1: Authenticate with Google Cloud
echo "==== Authenticating with Google Cloud ===="
gcloud auth configure-docker

# Step 2: Build and push Docker images with platform specifications
echo "==== Building and pushing Docker images ===="
echo "Setting up multi-platform builder..."
docker buildx create --name multiplatform-builder --use || echo "Builder already exists, continuing..."
docker buildx inspect --bootstrap

echo "Building and pushing API service image..."
docker buildx build --platform linux/amd64 \
  -t gcr.io/$PROJECT_ID/api-service:latest \
  -f Dockerfile.api . --push

echo "Building and pushing Worker service image..."
docker buildx build --platform linux/amd64 \
  -t gcr.io/$PROJECT_ID/worker-service:latest \
  -f Dockerfile.worker . --push

# Step 3: Check if cluster exists, create if not
echo "==== Setting up Kubernetes cluster ===="
if gcloud container clusters list --filter="name=$CLUSTER_NAME" --format="get(name)" | grep -q $CLUSTER_NAME; then
  echo "Cluster $CLUSTER_NAME already exists, getting credentials..."
  gcloud container clusters get-credentials $CLUSTER_NAME --zone=$REGION
else
  echo "Creating Kubernetes cluster $CLUSTER_NAME..."
  gcloud container clusters create $CLUSTER_NAME \
    --num-nodes=$NODE_COUNT \
    --zone=$REGION \
    --machine-type=$MACHINE_TYPE
fi

# Step 4: Install GKE auth plugin if needed
if ! command -v gke-gcloud-auth-plugin &> /dev/null; then
  echo "GKE auth plugin not found, installing..."
  if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    sudo apt-get install google-cloud-sdk-gke-gcloud-auth-plugin
  elif [[ "$OSTYPE" == "darwin"* ]]; then
    brew install --cask google-cloud-sdk
  else
    echo "Please install gke-gcloud-auth-plugin manually following: https://cloud.google.com/kubernetes-engine/docs/how-to/cluster-access-for-kubectl#install_plugin"
  fi
  
  # Refresh credentials after plugin installation
  gcloud container clusters get-credentials $CLUSTER_NAME --zone=$REGION
fi

# Step 5: Create or apply Kubernetes manifests
echo "==== Applying Kubernetes manifests ===="

# Redis deployment
cat > redis-deployment.yaml << EOF
apiVersion: apps/v1
kind: Deployment
metadata:
  name: redis
spec:
  selector:
    matchLabels:
      app: redis
  template:
    metadata:
      labels:
        app: redis
    spec:
      containers:
      - name: redis
        image: redis:latest
        ports:
        - containerPort: 6379
---
apiVersion: v1
kind: Service
metadata:
  name: redis
spec:
  selector:
    app: redis
  ports:
  - port: 6379
    targetPort: 6379
EOF

# API deployment
cat > api-deployment.yaml << EOF
apiVersion: apps/v1
kind: Deployment
metadata:
  name: api-service
spec:
  replicas: 2
  selector:
    matchLabels:
      app: api-service
  template:
    metadata:
      labels:
        app: api-service
    spec:
      containers:
      - name: api
        image: gcr.io/$PROJECT_ID/api-service:latest
        ports:
        - containerPort: 8080
        env:
        - name: REDIS_HOST
          value: "redis"
        - name: REDIS_PORT
          value: "6379"
        - name: PORT
          value: "8080"
---
apiVersion: v1
kind: Service
metadata:
  name: api-service
spec:
  type: LoadBalancer
  selector:
    app: api-service
  ports:
  - port: 80
    targetPort: 8080
EOF

# Worker deployment
cat > worker-deployment.yaml << EOF
apiVersion: apps/v1
kind: Deployment
metadata:
  name: worker-service
spec:
  replicas: 5
  selector:
    matchLabels:
      app: worker-service
  template:
    metadata:
      labels:
        app: worker-service
    spec:
      containers:
      - name: worker
        image: gcr.io/$PROJECT_ID/worker-service:latest
        env:
        - name: REDIS_HOST
          value: "redis"
        - name: REDIS_PORT
          value: "6379"
EOF

# Apply the manifests
echo "Applying Redis deployment..."
kubectl apply -f redis-deployment.yaml

echo "Applying API service deployment..."
kubectl apply -f api-deployment.yaml

echo "Applying Worker service deployment..."
kubectl apply -f worker-deployment.yaml

# Step 6: Wait for deployments to be ready
echo "==== Waiting for deployments to be ready ===="
kubectl rollout status deployment/redis
kubectl rollout status deployment/api-service
kubectl rollout status deployment/worker-service

# Step 7: Get the external IP
echo "==== Getting service information ===="
echo "Waiting for external IP assignment (may take a minute)..."
while [ -z "$(kubectl get service api-service -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null)" ]; do
  echo -n "."
  sleep 5
done
echo ""

EXTERNAL_IP=$(kubectl get service api-service -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
echo "==== Deployment complete! ===="
echo "Your API is available at: http://$EXTERNAL_IP/"
echo "Redis is running as an internal service at: redis:6379"
echo "Worker service is running with 5 replicas"