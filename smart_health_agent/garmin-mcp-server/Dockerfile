# Multi-stage Dockerfile for Garmin MCP Server
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Create non-root user
RUN groupadd --gid 1000 mcpuser && \
    useradd --uid 1000 --gid mcpuser --shell /bin/bash --create-home mcpuser

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Development stage
FROM base as development

# Install development dependencies
RUN pip install --upgrade pip

# Set work directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt requirements-dev.txt ./
RUN pip install -r requirements-dev.txt

# Copy source code
COPY . .

# Install package in development mode
RUN pip install -e .

# Change to non-root user
USER mcpuser

# Default command for development
CMD ["python", "-m", "garmin_mcp.server"]

# Production stage
FROM base as production

# Install production dependencies only
RUN pip install --upgrade pip

# Set work directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY pyproject.toml README.md LICENSE ./

# Install package
RUN pip install .

# Create directories for tokens and logs
RUN mkdir -p /app/data /app/logs && \
    chown -R mcpuser:mcpuser /app

# Change to non-root user
USER mcpuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# Expose port (if running HTTP transport)
EXPOSE 8000

# Default command
CMD ["python", "-m", "garmin_mcp.server"]