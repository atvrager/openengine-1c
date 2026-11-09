# Containerfile for OpenEngine-1C Native Solver Suite
# Packages: OpenFOAM v2406, CalculiX (ccx), Gmsh
FROM docker.io/opencfd/openfoam-default:2406

USER root
ENV DEBIAN_FRONTEND=noninteractive

# Install CalculiX (ccx), Gmsh, and supporting runtime libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    calculix-ccx \
    gmsh \
    libgomp1 \
    libglu1-mesa \
    libgl1-mesa-glx \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Set up working directory
WORKDIR /workspace

# Default entrypoint activates OpenFOAM environment
ENTRYPOINT ["/bin/bash", "-c"]
CMD ["echo 'OpenEngine Solvers Ready (OpenFOAM 2406, CalculiX, Gmsh)'"]
