FROM python:3.9.17-bookworm
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgstreamer1.0-0 \
    libgstreamer-plugins-base1.0-0 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*
ENV PYTHONBUFFERED True
ENV APP_HOME /back-end
# Set the working directory in the container
WORKDIR $APP_HOME

# Copy the current directory contents into the container at /app
COPY . ./

# Install any needed packages specified in requirements.txt
# If you don't have a requirements.txt, remove this RUN command

RUN pip install --no-cache-dir -r requirements.txt


# Run the application
# Replace 'app.py' with your main application file
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 app:app