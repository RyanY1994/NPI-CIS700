# NPI-CIS700

#### Overview
Here is the final project program of CIS700 Neural Program Learning Course.

The objective of this program is apply Neural Programmer Interpreters (NPIs) for the task to distinguish all non-zero elements in a matrix and mark(reverse) them according to their original value.

#### File Functions
train.py: Establish and train the NPIs model for the task

test.py: Load and test the trained NPIs model for the task

trace.py: Generate execution traces (Input and Output/Label) required by the training process of NPIs model

#### Requirements
Python 3.5

Tensorflow 1.4.0

CUDA 7.5

#### Usage
If you want to train the model, simply run the code, and the trained model will be saved in the ./checkpoints/ folder
```
python3 train.py
```

If you want to check the losses during training process, then run the command and enter 'localhost:6006' use your browser
```
tensorboard --logdir=Tensorboard
```

If you want test the model using a specified matrix and show the generated commands, run the code, it will load the latedst model in the ./checkpoints/ folder
```
python3 test.py --TestType single --ShowCommands
```

If you want test the model with multiple randomly generated matrix samples and do not show the generated commands, run the code
```
python3 test.py --TestType multiple
```
