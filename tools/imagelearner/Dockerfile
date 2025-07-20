# Tell Docker to pull amd64 variant on your arm64 host (optional)
FROM --platform=linux/amd64 quay.io/goeckslab/galaxy-ludwig-gpu:0.10.1

# Switch to root (you’re already root by default, but no harm)
USER root

# Install plotly
RUN pip install --no-cache-dir plotly
