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
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Ensure OpenFOAM environment is sourced in all non-login and login shells
RUN echo "source /usr/lib/openfoam/openfoam2406/etc/bashrc" >> /etc/bash.bashrc

# Set up working directory
WORKDIR /workspace

ENTRYPOINT ["/bin/bash", "-c"]
CMD ["echo 'OpenEngine Solvers Ready (OpenFOAM 2406, CalculiX ccx, Gmsh)'"]
