#!/bin/bash

##################################################################DOCKER SETUP################################################################################

# Define the Docker version to install
DOCKER_VERSION="24.0.5"

# Update the apt package index
sudo apt-get update

# Check if Docker is installed
if ! [ -x "$(command -v docker)" ]; then
    # If Docker is not installed, download and install it
    echo "Docker is not installed. Installing..."
    curl -fsSL https://download.docker.com/linux/static/stable/x86_64/docker-$DOCKER_VERSION.tgz | sudo tar -xzC /usr/local/bin --strip-components=1
else
    # Check if the installed Docker version is different from the desired version
    if [ "$(docker --version | awk '{print $3}' | tr -d ',')" != "$DOCKER_VERSION" ]; then
        # If the versions are different, upgrade Docker
        echo "Upgrading Docker to version $DOCKER_VERSION..."
        sudo rm -f /usr/local/bin/docker
        curl -fsSL https://download.docker.com/linux/static/stable/x86_64/docker-$DOCKER_VERSION.tgz | sudo tar -xzC /usr/local/bin --strip-components=1
    else
        # If the versions are the same, just print a message
        echo "Docker version $DOCKER_VERSION is already installed."
    fi
fi

# Print a success message
echo "Using Docker version $DOCKER_VERSION"


##################################################################GOOGLE CLOUD SETUP################################################################################

# Download the Google Cloud SDK
curl https://sdk.cloud.google.com | bash

# Initialize the Google Cloud environment
gcloud init

# Login using the service account
gcloud auth login

# Activate the service account
gcloud auth activate-service-account --key-file="" # Add the path to the service account key file

# Configure Docker to use gcloud as a credential helper
gcloud auth configure-docker us-west2-docker.pkg.dev