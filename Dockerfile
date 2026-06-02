FROM continuumio/miniconda3

WORKDIR /app

COPY environment.yml .

RUN conda env create -f environment.yml && conda clean -afy

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "msf_py3", "python"]