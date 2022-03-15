# FMMC

## Docker image

To build docker image from file, run:

```
docker build --no-cache -t fmmc --build-arg user=<Your_USERNAME> -f Dockerfile .
```

This command creates a new docker image with the name 'fmmc'. The '.' indicates that the Dockerfile is in the current directory.

To run this image as a container:
```
docker run --gpus all -it --rm -v $HOME/path/to/FMMC:/work -w /work fmmc
```
