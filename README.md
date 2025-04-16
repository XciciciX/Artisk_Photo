# Artisk_Photo

### Cors
gcloud storage buckets update gs://run-sources-artisk-photo-us-east1 --cors-file=cors.json


### Gcloud 
gcloud auth login

gcloud run deploy artiskphoto \ 
    --source . \
    --region us-east1 \
    --allow-unauthenticated
    
gcloud run deploy artiskphoto-api \
    --image gcr.io/artisk-photo/artiskphoto-api \
    --region us-east1 \
    --allow-unauthenticated \
    --set-env-vars REDIS_HOST=redis,REDIS_PORT=6379

gcloud run deploy artiskphoto-worker\
    --image gcr.io/artisk-photo/artiskphoto-worker \
    --region us-east1 \
    --allow-unauthenticated \
    --set-env-vars REDIS_HOST=redis,REDIS_PORT=6379

gcloud builds submit --tag gcr.io/artiskphoto/artiskphoto-api


docker-compose down
docker-compose up -d --build


I'll guide you through testing the asynchronous image processing service locally. Here are the steps to get everything up and running and test it properly:
Setting Up and Testing Locally
Step 1: Start the Services
First, make sure you have your GEMINI_API_KEY set:
bashexport GEMINI_API_KEY=your_api_key
Then start the Docker containers:
bashdocker-compose up -d
Verify all containers are running:
bashdocker-compose ps
You should see the Redis, API, and worker services running.
Step 2: Test the Health Endpoint
First, check if the API service is operational:
bashcurl http://localhost:8080/health
You should receive a response like:
json{
  "status": "healthy",
  "redis": "connected"
}
Step 3: Submit an Image for Processing
Upload an image for processing:
bashcurl -X POST -F "image=@/path/to/your/image.png" -F "max_iterations=5" -F "initial_threshold=150" http://localhost:8080/process
Replace /path/to/your/image.png with the actual path to a test image on your machine.
The API should return something like:
json{
  "status": "success",
  "message": "Image processing job submitted",
  "session_id": "1a2b3c4d-5e6f-7g8h-9i0j-1k2l3m4n5o6p"
}
Save the session_id for the next steps.
Step 4: Check Job Status
Check the status of your job:
bashcurl http://localhost:8080/status/1a2b3c4d-5e6f-7g8h-9i0j-1k2l3m4n5o6p
Replace the ID with your actual session ID. You'll see responses like:
json{
  "status": "pending",
  "message": "Job is still processing"
}
Or when processing:
json{
  "status": "processing",
  "message": "Job is still processing"
}
Step 5: Get the Results
Once the job is completed, get the results:
bashcurl http://localhost:8080/result/1a2b3c4d-5e6f-7g8h-9i0j-1k2l3m4n5o6p
Again, use your actual session ID. You should receive a response containing base64-encoded images:
json{
  "status": "completed",
  "final_threshold": 140,
  "acceptable": true,
  "iterations": 3,
  "bw_image_base64": "base64_encoded_string...",
  "inverted_image_base64": "base64_encoded_string..."
}
Step 6: Save the Images Locally
To view the processed images, you can save the base64 strings to files:
bash# Create a script to decode and save the image
echo 'import sys, json, base64

    f.write(base64.b64decode(data["inverted_image_base64"]))
print("Images saved as bw_image.png and inverted_image.png")' > decode_images.py

# Save the API response to a file
curl http://localhost:8080/result/1a2b3c4d-5e6f-7g8h-9i0j-1k2l3m4n5o6p > result.json

# Run the script to extract and save images
python decode_images.py result.json
Step 7: Monitor Container Logs
To see what's happening behind the scenes:
bash# API service logs
docker-compose logs -f api

# Worker service logs
docker-compose logs -f worker
This is useful for debugging if something isn't working as expected.
Step 8: Testing Redis Queue
You can also directly inspect the Redis queue to make sure jobs are being added and processed:
bash# Enter Redis CLI
docker-compose exec redis redis-cli

# Check queue length
LLEN image_processing_queue

# Check for job details
KEYS job:*

# Get a specific job's details
GET job:1a2b3c4d-5e6f-7g8h-9i0j-1k2l3m4n5o6p
Step 9: Testing Scaling
Try processing multiple images simultaneously and see how the workers handle the load:
bash# Scale up workers
docker-compose up -d --scale worker=5

# Submit multiple jobs at once (run in parallel)
curl -X POST -F "image=@./test/test06.jpg" http://34.122.242.153/process & 
curl -X POST -F "image=@/path/to/image2.png" http://localhost:8080/process &
curl -X POST -F "image=@/path/to/image3.png" http://localhost:8080/process &
Step 10: Clean Up
When done testing:
bash# Stop all services
docker-compose down

# Optionally remove volumes
docker-compose down -v
Troubleshooting
If you encounter issues:

Images not processing: Check worker logs to see if there are any errors in the Gemini API calls or image processingwith open(sys.argv[1]) as f:
    data = json.load(f)
with open("bw_image.png", "wb") as f:
    f.write(base64.b64decode(data["bw_image_base64"]))
with open("inverted_image.png", "wb") as f:
Connection issues: Verify Redis is running and accessible to both the API and worker services
Permission problems: Make sure shared volumes have the right permissions
Missing dependencies: In case of missing libraries in the containers, you may need to modify the Dockerfiles and rebuild with docker-compose build

By following these steps, you should be able to fully test the asynchronous image processing service locally.