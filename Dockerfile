FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends sox libsox-fmt-all && rm -rf /var/lib/apt/lists/*

# Install pip packages - either from local wheels or from requirements.txt
# To run fully offline, place .whl files in wheels/ and uncomment the next two lines:
# COPY wheels/ /app/wheels/
# RUN pip install --no-cache-dir --no-index --find-links /app/wheels/ -r requirements.txt

# Online install (comment out when using local wheels)
COPY requirements.txt .
RUN pip install --no-cache-dir numpy typing_extensions && pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY voices.yaml .

# Reference audio files for voice cloning (if bundled into image)
# To use external voices, mount them: -v /path/to/voices:/app/voices
COPY voices/ /app/voices/

# Models should be mounted or copied into /app/models/
# e.g. docker run -v /path/to/models:/app/models ...
# or uncomment to bake them in:
# COPY models/ /app/models/

ENV MODEL_PATH_0_6B=/app/models/Qwen3-TTS-12Hz-0.6B-Base
ENV MODEL_PATH_1_6B=/app/models/Qwen3-TTS-25Hz-1.5B-Base
ENV VOICES_CONFIG=/app/voices.yaml

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
