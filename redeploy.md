if i change the code, do i need to repush docker images

Edit
Yes, if you make changes to your code, you'll need to rebuild and repush your Docker images for those changes to take effect. Here's the process:

Make your code changes locally
Rebuild the Docker images:
bash
docker build -t gcr.io/artisk-photo/api-service:v1 -f Dockerfile.api .
docker build -t gcr.io/artisk-photo/worker-service:v1 -f Dockerfile.worker .
Push the updated images to Google Container Registry:
bash
docker push gcr.io/artisk-photo/api-service:v1
docker push gcr.io/artisk-photo/worker-service:v1
Restart the deployments on your Kubernetes cluster:
bash
kubectl rollout restart deployment api-service
kubectl rollout restart deployment worker-service
However, if you're just testing locally, you can avoid rebuilding for every small change by using volume mounts in your docker-compose.yml:


docker buildx build --platform linux/amd64 -t gcr.io/artisk-photo/worker-service:v1 .
docker push gcr.io/artisk-photo/worker-service:v1

kubectl set image deployment/worker-service worker=gcr.io/artisk-photo/worker-service:v1

kubectl rollout restart deployment/worker-service

# Build and push the API image with the correct platform
docker buildx build --platform linux/amd64 -t gcr.io/artisk-photo/api-service:v1 .
docker push gcr.io/artisk-photo/api-service:v1

# Create or update the API deployment
kubectl set image deployment/api-service api=gcr.io/artisk-photo/api-service:v1

# Restart the API deployment
kubectl rollout restart deployment/api-service

kubectl delete deployment worker-service
kubectl delete deployment api-service
kubectl delete deployment redis