# Adthena Test

## Project Description
This project uses a Support Vector Machine (Linear_SVC) model to categorize search terms into industry categories.
The model is written in Python and includes files which pre-process the raw data, train the model and then predict categories in a test dataset. 

### Getting Setup
##### 1. Install Python 3.6.2

```
pyenv install 3.6.2
```

##### 2. Clone this repo in to your local directory. You can do this from github or use the following code:
```
git clone git@github.com:lebatchelor/adthena_lb_test.git
```

##### 3. Create a python virtual environment
```
cd ~adthena-lb-test
virtualenv -p ~/.pyenv/versions/3.6.2/bin/python --no-site-packages py
```

##### 4. Activate the virtual python environment and install python packages with pip
```
source py/bin/activate
pip install -r requirements.txt
```

### Running the Model
##### 1. Train and save the model for future use.
<trainSet.csv> can be replaced with any other csv file path to training data. 
```
cd ~adthena-lb-test
source py/bin/activate
python model/run.py data/trainSet.csv train

```
##### 1. To pre-process training data and train the model, run these commands from your command line. 
<candidateTestSet.csv> can be replaced with any other csv or txt file path with test data. 
The program will detect whether test set y data has been provided in the file and produce stats if so.
```
cd ~adthena-lb-test
source py/bin/activate
python model/run.py data/candidateTestSet.csv test

```

---

**Contributers:** Laura Batchelor - laura.batchelor@adthena.com
