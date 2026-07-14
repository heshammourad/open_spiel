# Stage 1: Build OpenSpiel and Python virtual environment
FROM python:3.11-bookworm as builder

ARG DEBIAN_FRONTEND=noninteractive
ARG NPROC=4

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    clang \
    git \
    curl \
    sudo \
    python3-dev \
    python3-pip \
    python3-setuptools \
    python3-wheel \
    python3-tk \
    python3-venv \
    virtualenv \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set compiler to clang
ENV CC=clang
ENV CXX=clang++

WORKDIR /app
COPY . .

# Initialize and install requirements
RUN chmod +x ./install.sh && ./install.sh
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Compile OpenSpiel C++ core and python bindings
RUN mkdir build && cd build \
    && cmake -DPython3_EXECUTABLE=$(which python) -DCMAKE_CXX_COMPILER=clang++ ../open_spiel \
    && make -j${NPROC}

# Stage 2: Runtime image
FROM python:3.11-slim-bookworm as runner

WORKDIR /app

# Copy python dependencies and build directory from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /app /app

# Set PYTHONPATH for the compiled pyspiel and open_spiel python package
ENV PYTHONPATH="/app:/app/build/python"

# Expose Google Cloud Run default port
EXPOSE 8080

# Start Streamlit playroom
ENTRYPOINT ["streamlit", "run", "azul_web_app.py", "--server.port", "8080", "--server.address", "0.0.0.0"]
