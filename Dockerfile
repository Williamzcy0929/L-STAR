# Dockerfile for L-STAR with preinstalled Palo R package
FROM rocker/r-ver:4.3.0

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    libcurl4-openssl-dev \
    libssl-dev \
    libxml2-dev \
    libgit2-dev \
    && rm -rf /var/lib/apt/lists/*

# Install R packages needed for Palo
RUN R -e "install.packages(c('remotes', 'devtools', 'RColorBrewer'), repos='https://cloud.r-project.org')"

# Install Palo from GitHub (pin to main branch, or use a specific commit/tag)
# Using remotes::install_github for better dependency handling
RUN R -e "remotes::install_github('Winnie09/Palo', repos='https://cloud.r-project.org')"

# Verify Palo installation
RUN R -q -e "library(Palo); cat('Palo version:', as.character(packageVersion('Palo')), '\n')"

# Set working directory
WORKDIR /app

# Copy Python requirements
COPY pyproject.toml ./

# Install Python dependencies
RUN pip3 install --no-cache-dir -e .

# Copy project files
COPY . .

# Default command
CMD ["python3", "-c", "import lstar; print('L-STAR ready')"]
