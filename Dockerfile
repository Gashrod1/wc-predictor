FROM python:3.11-slim

WORKDIR /app

# Create virtualenv inside image
RUN python -m venv /app/.venv

# Install dependencies into the venv
COPY requirements.txt .
RUN /app/.venv/bin/pip install --upgrade pip && \
    /app/.venv/bin/pip install -r requirements.txt

# Copy source code
COPY . .

# Always use venv Python
ENV PATH="/app/.venv/bin:$PATH"
ENV VIRTUAL_ENV="/app/.venv"

# Default command
CMD ["python", "cli.py", "--help"]
