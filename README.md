# user_engagement_regression

## Setup 
I tested it for Python 3.8.6. 
<br />
Create a virtual environment (preferably).
<br />
&emsp;&emsp;&emsp; 1. `pip install -r requirements.txt` <br />
&emsp;&emsp;&emsp; 2. `mkdir datasets; cd datasets` <br />
&emsp;&emsp;&emsp; 3. Put the CSV file in this directory <br />
&emsp;&emsp;&emsp; 4. `cd ..; mkdir models; cd models` <br />
&emsp;&emsp;&emsp; 5 The fine-tuned model can be found [here](https://drive.google.com/file/d/1-9pezTmcx486Exgo4Bo2eINRoqJNwdPz/view?usp=sharing). Download it and put in the models dir.


## Usage
### Create the train, val and test data (3 seperate csv files)
`python create_data.py`

### Training
`bash run_train.sh`
<br /><br />
It will create a `lightning_logs` directory in your current directory. Each run will have its own fine-tuned model and the tensorboard plots.

### Testing
Change the model path in `run_test.sh` after training or use it as it is to test on already fine-tuned model. 
`bash run_test.sh`

<br />
Note: All the terminal logs will in the `./logs` directory

## Take-aways
In my excitement, I immediately jumped on implementing the current best model and use it to fine-tune for my task. Later, on I realized that I don't have any benchmark to compare my results. <br />
Note: Test results can be found in 
