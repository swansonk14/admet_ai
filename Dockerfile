# Set the base image to use, which is "mambaorg/micromamba:1.5.1"
FROM mambaorg/micromamba:1.5.1

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
RUN micromamba install -y -n base -c conda-forge python=3.10 xorg-libxrender && \
    micromamba clean --all --yes

# Copy the current directory from the host to the container's "/opt/admet_ai" directory
COPY --chown=$MAMBA_USER:$MAMBA_USER . /opt/admet_ai

# Set the working directory to "/opt/admet_ai"
WORKDIR /opt/admet_ai

# Install the Python package in editable mode within the conda environment "base" previously created
# THen, clean up the pip cache
RUN /opt/conda/bin/python -m pip install -e .[web] && \
   /opt/conda/bin/python -m pip cache purge

# Expose port 5000 for the container to listen on
EXPOSE 5000

# Change the working directory to "/opt/admet_ai/admet_ai/web"
WORKDIR /opt/admet_ai/admet_ai/web

# Switch back to the root user to execute the final command
USER root

# The default command to run when the container starts.
# It uses Gunicorn to serve the application on 0.0.0.0:5000
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "wsgi:build_app()"]
