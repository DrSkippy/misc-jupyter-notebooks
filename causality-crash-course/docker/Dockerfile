# Optional image: the stock datascience notebook plus the extras some notebooks install on first use.
FROM quay.io/jupyter/datascience-notebook:latest

# R packages for causality-crash-course (ipw is not on conda-forge, so it comes from CRAN)
RUN mamba install --yes -c conda-forge \
        r-tableone r-matching r-matchit r-survey r-sandwich r-fixest r-geepack && \
    Rscript -e 'install.packages("ipw", repos = "https://cloud.r-project.org")' && \
    mamba clean --all -f -y && \
    fix-permissions "${CONDA_DIR}" && \
    fix-permissions "/home/${NB_USER}"

# CuPy with the CUDA 12 runtime libraries from pip, for time-to-traffic-metric's `backend: cupy|auto`.
# Only the host's NVIDIA driver is needed (plus the GPU block in compose.yaml).
RUN pip install --no-cache-dir "cupy-cuda12x[ctk]"
