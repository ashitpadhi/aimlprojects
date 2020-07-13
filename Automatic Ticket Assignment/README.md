# Intro

# Setup

```
pip install pipenv
pipenv shell
pipenv install
python -m spacy download en_core_web_sm
python -m spacy download en
```

### Test code coverage

`pytest --cov-report term-missing --cov=src tests/`

### Run all Unit Tests

`python -m unittest`

### Run Specific Unit Test

`python -m unittest tests.test_preprocessor_helpers.PreprocessorHelpersTest.test_clean_corpus`

# Deployment

## Helpful Links

### Pipenv

-   [Pipenv Guide](https://realpython.com/pipenv-guide/)
-   [pipenv basicss](https://pipenv-fork.readthedocs.io/en/latest/basics.html)
-   [Pipenv Cheat Sheet](https://gist.github.com/bradtraversy/c70a93d6536ed63786c434707b898d55)

### Miscellaneous

-   [textblob -- Detect Language](https://textblob.readthedocs.io/en/dev/quickstart.html)
-   [model training server -- cloud](https://www.floydhub.com/)
-   [Mlab for MongoDB](https://mlab.com/)
-   [Mlab MongoDB Examples](https://blog.mlab.com/2011/11/ample-mongodb-examples/)

## Preprocessor Usage

put input data file into data folder and provide its name into src.dataset before running
following command

```
python -m src.dataset_preprocessor
```

or run following test

```
python -m unittest tests.test_preprocessor_helpers.PreprocessorHelpersTest.test_clean_corpus
```
