# ai-label-trainer-service

A docker container for training an AI model using a neural network for labeling text.

Executes a job, defined by the [ai-job-service](https://github.com/redpencilio/ai-job-service)

## Getting started

```yml
services:
  label-trainer:
    image: redpencil/ai-label-trainer
    environment:
      LOG_LEVEL: "debug"
      TASK: "predicates"
    links:
      - db:database
    volumes:
      - ./config/label-trainer/constants.py:/app/constants.py
      - ./share:/share
  #    deploy:
  #      resources:
  #        limits:
  #          cpus: 4
```

The `TASK` variable is the type of task that starts this training loop using
the [job-run-loop](https://github.com/stijnrosaer/job-run-loop). This task should be used when creating a job with
the [ai-job-service](https://github.com/redpencilio/ai-job-service).

You probably need to define the amount of virtual cpu cores that this service allowed to use to limit usage. Note that
performance will also decrease with less resources.

If an Nvidia GPU is available, it is advised to change the [Dockerfile](Dockerfile) so it allows the use of a GPU.

After training, the job is updated with the id of the file containing the trained model, stored in the triplestore.

### Configuration

In `config/label-trainer/constants.py`:

```python
import torch

SEP_TOKEN = '[SEP]'
CLS_TOKEN = '[CLS]'
MODEL_FILE_PATH = '/share/model/predicate-model.pth'
BATCH_SIZE = 4
NUM_EPOCHS = 3
GRADIENT_ACCUMULATION_STEPS = 8
MAX_CLASS_SIZE = 500
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
```

The `MAX_CLASS_SIZE` defines the maximum amount of items per class that sould be used for training. Keep in mind that
biasness due to class imbalance can occur if the number of items per class are very different. A larger possibly
increases performance, but also learning time.

Make sure to mount this python file in the correct location of the container as shown in the docker-compose entry above!

## Tensorboard
Tensorboard was used to log training loss and accuracy in a dashboard. All data will be written to `share/tb/`.
In order to start up the dashboard, make sure tensorboard is installed locally with:
```
pip install tensorboard
```

starting tensorboard can now be done by running:
```
tensorboard --logdir=share/tb
```

Tensorboard is now available at `http://localhost:6006` if that port is available, otherwise see terminal for correct port.

## Reference

### Environment variables

- `LOG_LEVEL` takes the same options as defined in the
  Python [logging](https://docs.python.org/3/library/logging.html#logging-levels) module.


- `MU_SPARQL_ENDPOINT` is used to configure the SPARQL query endpoint.

    - By default this is set to `http://database:8890/sparql`. In that case the triple store used in the backend should
      be linked to the microservice container as `database`.


- `MU_SPARQL_UPDATEPOINT` is used to configure the SPARQL update endpoint.

    - By default this is set to `http://database:8890/sparql`. In that case the triple store used in the backend should
      be linked to the microservice container as `database`.


- `MU_APPLICATION_GRAPH` specifies the graph in the triple store the microservice will work in.

    - By default this is set to `http://mu.semte.ch/application`. The graph name can be used in the service
      via `settings.graph`.


- `MU_SPARQL_TIMEOUT` is used to configure the timeout (in seconds) for SPARQL queries.

## Improvements

Every time the model is trained, it is stored on disk under the file name, defined in the config.py file. This means
that if training is done two times, the original model can be overwritten. The link to this file in the triple store
will not be changed. So, the older model will no longer be available while the id for this one will give the new model.