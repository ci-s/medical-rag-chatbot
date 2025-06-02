# Use a slim Python image
FROM python:3.12-slim

# Set environment variables
ENV POETRY_VERSION=1.8.2 \
    POETRY_VIRTUALENVS_CREATE=false \
    PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    tesseract-ocr \
    poppler-utils \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
# RUN curl -sSL https://install.python-poetry.org | python3 - && \
#     ln -s ~/.local/bin/poetry /usr/local/bin/poetry

# Install uv
# Download the latest installer
ADD https://astral.sh/uv/install.sh /uv-installer.sh

# Run the installer then remove it
RUN sh /uv-installer.sh && rm /uv-installer.sh

# Ensure the installed binary is on the `PATH`
ENV PATH="/root/.local/bin/:$PATH"

# Set work directory
WORKDIR /app

# Copy poetry files first (for caching)
COPY pyproject.toml uv.lock* README.md ./

# Install Python dependencies
RUN uv sync
# RUN pip install gpt4all
# Copy the rest of the application code
COPY . .

# Expose port (assuming you run uvicorn on 8000)
EXPOSE 7001

ENV PYTHONPATH=/app/src
# Default command
# CMD ["uv", "run", "uvicorn", "src.serve.app:app", "--host", "0.0.0.0", "--port", "7001", "--reload"]