FROM python:3.11-slim

# 1. Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/home/user/app

# 2. Create a non-root user (Required by HF for safety)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# 3. Copy requirements and install
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# 4. Copy the rest of the application
COPY --chown=user . .

# 5. Make the shell script executable
RUN chmod +x run.sh

# 6. Expose the HF port
EXPOSE 7860

# 7. Start the combined processes
CMD ["./run.sh"]