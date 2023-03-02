# languagemodels

A simple toolkit to train and evaluate language models.

## Setup

### Recommended: Docker

- Build the Docker image. It is build from the official Nvidia PyTorch image (Versio 21.03) and already contains Python 3.8, miniconda and CUDA.

```shell
docker build \
     -f ./Dockerfile \
     --build-arg USER_UID=$UID \
     --build-arg USER_NAME=$(id -un) \
     -t languagemodels:latest .
```

- Run the container from the project folder. You still need to change the paths to the actual paths on your machine.

```shell
export USER_NAME=$(id -un)
docker run -it --rm --runtime=nvidia --pid=host --ipc=host --user $USER_NAME \
    -v /nethome/$USER_NAME/git/lsv/languagemodels:/languagemodels \
    -v /data/users/$USER_NAME/pre-trained-transformers:/pre-trained-transformers \
    -v /data/users/$USER_NAME/datasets:/datasets \
    -v /data/users/$USER_NAME/logs/languagemodels/logfiles:/logfiles \
    languagemodels:latest
```

- Running the container in this way will install the `languagemodels` package in development mode. This enables changing the code of the package on-the-fly without rebuilding the container.
- You can leave a running container via: `CTRL + p + q`
- You can re-attach to a container by running: `docker attach <container-name>`

### Python venv

- Create a new virtual environment: `python3 -m venv ./languagemodels-venv`
- Activate the virtual environment: `source languagemodels-venv/bin/activate`
- Install package & requirements: `pip install -e .`

### Python miniconda

- Create a new virtual environment: `conda create --name languagemodels python=3.7`
- Activate the virtual environment: `conda activate languagemodels`

- Upgrade pip: `pip install --upgrade pip`
- Install package & requirements: `pip install -e .`

### Pytorch

- If you do not go with the Docker (recommended) method, install the appropriate version of Pytorch before installing the package: https://pytorch.org/get-started/locally/ 

### Running scripts

- Once the virtual environment is activated and the package has been installed, scripts from `bin/` can be run from the command line using `<name of script>`, e.g.: `eval_lm -h`

## Contributing

| Contributor    | Email                        |
|:---------------|:-----------------------------|
| Marius Mosbach | mmosbach@lsv.uni-saarland.de |
| Julius Steuer  | jsteuer@lsv.uni-saarland.de  |