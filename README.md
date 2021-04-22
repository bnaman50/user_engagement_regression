# user_engagement_regression

## Problem Description
Trainig data can be found [here](https://drive.google.com/file/d/15X00ZWBjla7qGOIW33j8865QdF89IyAk/view). The dataset is tabular and the features involved should be self-explanatory. The problem should be treated as large-scale, as the dataset is large (e.g., >100GB) and will not fit into the RAM of your machine.
<br /><br />
I chose to implement the user-engagement prediction. Curently, I consider number of upvotes as the user engagement since downvotes are all zeros but it is easily extensible using my current steup by using (`--num_labels=2`).


## Setup 
I tested it for Python 3.8.6. 
<br />
Create a virtual environment (preferably).
<br />
1. `pip install -r requirements.txt` <br />
2. `mkdir datasets; cd datasets` <br />
3. Put the CSV file in this directory <br />
4. `cd ..; mkdir models; cd models` <br />
5. The fine-tuned model can be found [here](https://drive.google.com/file/d/1-9pezTmcx486Exgo4Bo2eINRoqJNwdPz/view?usp=sharing). Download it and put in the models dir.


## Usage
### Create the train, val and test data (3 seperate csv files)
`python create_data.py`
<br />
Note: This is just to ease the training process. I directly load the file into the memroy in this case. I was more focused on model training with huge file. 

### Training
`bash run_train.sh`
<br /><br />
It will create a `lightning_logs` directory in your current directory. Each run will have its own fine-tuned model and the tensorboard plots. You can also check the path of the best model in the logs file

### Testing
Change the model path in `run_test.sh` after training or use it as it is to test on already fine-tuned model. 
`bash run_test.sh`

<br />
Note: All the terminal logs will in the `./logs` directory

## Take-aways
In my excitement, I immediately jumped on implementing the current best model (I use DistilBert. Can easily be swapped with Bert) and use it to fine-tune for my task. Later, on I realized that I don't have any benchmark to compare my results. <br />
Note: Test results can be found in [here](./logs/test_results.txt). I don't have any baseline results to gauge my test results but I believe they are not good rn. 

## Possible Solutions and Future
1. Implement a simple TF-IDF based solution to be used as a baseline.
2. In my current setup, I am clipping the gradient norm at 1 which is considered to be a good practice. Both the training and validation loss are going down, but it is very slow and requires a lot more training. <br /> The following are the training and validation curves for 5 epochs. ![image](https://user-images.githubusercontent.com/5251592/115666003-a1d42480-a309-11eb-8c7f-5448fe0ec598.png) <br /> The following are the training and validation curves for 50 epochs (it is still running and taking way too much time given my limited resources). **The main point is that there is huge improvement here (2.9e5 --> 2.8e5) but I need to consider some form of normalization as mentioned in the next point.**
![image](https://user-images.githubusercontent.com/5251592/115666056-b57f8b00-a309-11eb-92dd-26875ef25eee.png)

3. Another possible reason for slow training (which is accentuated by previous reason) is that I am directly regressing on the raw user engagement numbers. There is a huge variance in upvotes values with some values being in thousands. Thus, I would need to have some sort of normalization but it is not directly clear what it should be since there in no pre-defined range. 
4. Implementing a data-loader for huge file was new thing to me. For the time being, I implemenetd a simple solution which uses the PyTorch's 'map-style' dataset. But it is slow. Thus, I would try to implement 'iterable-style' dataset (I tried it out but there were some issues such as duplicate data on multiple workers, not being able to randomize the training data etc.)
5. Right now, my model used news tilte as the input but I think other features like user_name (assumption: correlation between popular user and respective engagement) or time. 


