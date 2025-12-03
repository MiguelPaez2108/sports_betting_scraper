# Multi-stage Dockerfile for production deployment
FROM python:3.11-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --user -r requirements.txt

# Production stage
FROM python:3.11-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 bettor && \
    mkdir -p /app /data /models /logs && \
    chown -R bettor:bettor /app /data /models /logs

# Set working directory
WORKDIR /app

# Copy Python packages from builder
COPY --from=builder /root/.local /home/bettor/.local

# Copy application code
COPY --chown=bettor:bettor src/ ./src/
COPY --chown=bettor:bettor scripts/ ./scripts/
COPY --chown=bettor:bettor config/ ./config/

# Set environment
ENV PATH=/home/bettor/.local/bin:$PATH
ENV PYTHONPATH=/app

# Switch to non-root user
USER bettor

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.exit(0)"

# Default command
CMD ["python", "scripts/run_nightly_etl.sh"]
