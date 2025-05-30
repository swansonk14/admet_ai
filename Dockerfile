# Set the base image to use, which is "mambaorg/micromamba:1.5.1"
FROM mambaorg/micromamba:2.1.1

# Create a named volume at "/opt/admet_ai/data" in the container to persist data across container runs
VOLUME /opt/admet_ai/data

# Switch to the root user to perform some installation tasks
USER root

# Update the package list and install "git" package, then remove unnecessary cached files to reduce image size
RUN apt-get update && \
    apt-get install -y git && \
    rm -rf /var/lib/{apt,dpkg,cache,log}

# Switch back to the non-root user specified in the "MAMBA_USER" environment variable
USER $MAMBA_USER

# Create a new conda environment named "base"
# Then, clean up the micromamba cache and other unnecessary files
RUN micromamba install -y -n base -c conda-forge python=3.12 rdkit && micromamba clean --all --yes

# Set the working directory to "/opt/admet_ai"
WORKDIR /opt/admet_ai

# Install the Python package in editable mode within the conda environment "base" previously created
# THen, clean up the pip cache
RUN /opt/conda/bin/pip install git+https://github.com/sunghunbae/admet_ai.git && \
/opt/conda/bin/pip cache purge
