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

## Fairness Maximization via Mode Connectivity

Fairness Maximization via Mode Connectivity (FMMC) has two layers of abstraction: 

- **The outer action layer** includes **critical neighbourhood exploration via mode connectivity**, followed by a **multi-objective optimization** to achieve the objective specified in the inner layer. 
- **The inner objective layer** includes an objective we try to optimize. In this work we consider an optimization over fairness formulation, although the inner layer may regulate other objectives besides fairness. 

### Critical Neighbourhood Exploration via Mode Connectivity

This stage is based on [Mode Connectivity](https://github.com/timgaripov/dnn-mode-connectivity).

To create the set of candidate models, please, follow curve finding procedure, described in ][here](https://github.com/timgaripov/dnn-mode-connectivity#curve-finding). The final step ["Evaluating the curves"]("https://github.com/timgaripov/dnn-mode-connectivity#evaluating-the-curves") will create a .csv file of the form <DATASET_NAME>_Subject_<NUMBER> in the provided direction (--dir argument).

### Multi-Objective optimization

To find a pareto-optimal solution run
```
python find_best_point.py --dir=<LOCATION_TO_CSV> --beta=<beta_value> --subject=<subject_number> --dataset=<Tamil|ActRec> 

```

This will print average accuracy, evenness index, and F_beta for 3 points: start and end points of a curve, as well as the pareto-optimal point.


