## State-of-the-Art NLU Engine for intent classification and entity recognition using RASA 3.0.6  conversational AI platform

The NLU engine built using RASA opens ource platform is highly flexible and customizable.  You can create your own NLP pipeline as per your need. The NLU engine runs on top of open source RASA NLU core that provides range of configuration that meet most of your NLP needs. You can take a look at how to configure your own pipeline [here][url1]

Based on my experience DIET classifier seems to be much faster to train, accurate and serves to find both intent and entities.

Here is how you could customize your pipeline and use this NLP engine that can act as backbone to your chatbot engine.

## Installation
### Prerequisites
This was implemented and tested on windows 10 – (Windows Subsystem for Linux) but should also work on Linux and Mac OS X.
- Make sure docker, and docker-compose are installed on your server. 
- Detailed instructions can be found in the Docker [documentation][url2].
### Download
- Clone or download this repo. Unzip and copy files from repo to your preferred location
### Docker build and run
- Build the docker by running following command at root of your folder
```sh
$ docker build . -t nlu:latest
```
- Mount the local folder and run the docker (commands are different for mac/linux and windoes users)
```sh
# for mac/linux
$ docker run -it -p 5001:5001 -v $(pwd)/server:/root/server --name nlu nlu:latest python app.py
# for windows user
$ docker run -it -p 5001:5001 -v %cd%/server:/root/server --name nlu nlu:latest python app.py
```
- If your installation is successful, you should see the running https server with default port of 5001. (ignore uvicorn logging error messages)
- Since you have mounted local folder , you can make all the changes locally. You need to stop and remove the "nlu" container before you rerun the docker command with your new changes. ( Note- I had freezed terminal and had to restart the docker from docker desktop. Everythin worked fine after restart !)
```sh
$ docker stop nlu
$ docker rm nlu
$ docker run -it -p 5001:5001 -v $(pwd)/server:/root/server --name nlu nlu:latest python app.py
```
- You can now train the model and predict the intent using REST API. Refer below section for REST APIs.

## Usage
### Setup
- NLU Engine uses Rasa Open Source that performs intent classification along with entity recognition.
- You need to provide training data in yaml format. More details can be found in the [documentation][url3]. 
- You can create your own training data and upload it in /server/training_data/intents folder.
- Follow the naming convention domain_locale.yml where domain is name of function for which you are building your bot. (e.g. travel, hr, finance etc.) and locale is language. e.g. "banking_en.yml" These are used while making REST API call. (explained below)
- Here is a sample training data provided for retail banking scenario for demonstration. (see "/server/training_data/intents/banking_en.md" for more details)
```
version: "3.0"
nlu:

- intent: askTransferCharge
  examples: |
    - Will I be charged for transferring money
    - do transfers cost something?
    - is there a transfer charge?
    - Is there a charge
    - will i be charged for a transaction?
    - do xfers cost something?
    - is there a transfer fee
    - is there a xfer fee
    - how much is the transfer fee

- intent: checkBalance
  examples: |
    - What is balance in my account?
    - How much money is on my account?
    - What's left on that account?
    - How much do I have on that account?
    - What's the balance on that account?
    - How much money is left on that account?
    - what is my account balance
    - what's my account balance?
    - what's my account balance
    - what's my balance?
    - whats my account balance
    - balance in my account
```
- You can configure your own pipeline based on your preference (SPACY, BERT or DIET) and store it in folder "/server/core/config" folder. While creating a pipeline you need to select following components -
    *	Word Vector Sources (Mitie, Spacy, HFTransformers etc.)
    *	Tokenizers (Whitespace, Jieba, Mitie, Spacy, ConveRT etc.)
    *	Text Featurizer (Mitie, Spacy, ConveRT, RegEx, CountVector, LexicalSyntactic etc.)
    *	Intent Classifier (Mitie, SKlearn, BERT, DIET etc.)
    *	Entity Extractor (Mitie, Spacy, Entity Synonym, CRF, Duckling, DIET etc) 
- You can get more details on how to configure in this [documentation][url4]. 
Here is a sample config file that has worked well for given example
```
language: en
pipeline:
    - name: WhitespaceTokenizer
    - name: RegexFeaturizer
    - name: LexicalSyntacticFeaturizer
    - name: CountVectorsFeaturizer
    - name: CountVectorsFeaturizer
      analyzer: char_wb
      min_ngram: 1
      max_ngram: 4
    - name: DIETClassifier
      epochs: 100
    - name: EntitySynonymMapper
    - name: ResponseSelector
      epochs: 100
```
- Now here is a last step before you are ready to run the your NLU engine. You need to configure NLU properties file. Edit the “nlp.propoerties” file located at "/server/config" file
Line -24 => Select algorithm = NLU in  [NLP_ALGORITHM] section as you are using RASA NLU core as your underneath engine for intent classification.
Line -28 => add CONFIG_FILE parameter with your configuration file that you created in step 4 above

#### Training NLP Engine
*Important* – For any changes that you do in any of the files, you need to build and run the docker.
- Now if you have built and your container running , it’s time to train your NLP Engine on training data that you created using REST API
    * Method – POST
    * URL https://127.0.0.1:5001/train?domain=banking&locale=en   (assuming your domain is banking and language English)
    * curl command
    ```
    $ curl --insecure --location --request POST 'https://127.0.0.1:5001/train?domain=banking&locale=en'
    ```
    * Response - 
```    
{"messageId": "TRAIN_SUCCESS",
"domain": "banking",
"locale": "en",
"message": "{\"intents\": \"15\", \"utterances\": \"216\", \"model\": \"NLU:DIET\"}",
"model": "ENSEMBLE"
}
```
>Note – Training may take some time, have some patience

#### Intent Classification/ Entity Recognition
- If your training is successful, you can pass on utterance to classify its intent and get the entities if any.
    * Method – POST
    * URL - https://127.0.0.1:5001/predict?domain=banking&locale=en&userUtterance=I want to pay my SBI card    (assuming you domain is banking and language English)
    * curl command
    ```
    $ curl --insecure --location --request POST 'https://127.0.0.1:5001/predict?domain=banking&locale=en&userUtterance=I%20want%20to%20pay%20my%20SBI%20card'
    ```
    * Response - 
```
{"messageId": "PREDICT",
"domain": "banking",
"locale": "en",
"userUtterance": "I want to pay my SBI card",
"model": "ENSEMBLE",
"message": "{\"intent\": {\"name\": \"makePayment\", \"confidence\": \"0.50\"}, \"entities\": [{\"entity\": \"creditCard\", \"start\": 17, \"end\": 20, \"confidence_entity\": \"0.999\", \"value\": \"SBI\", \"extractor\": \"DIETClassifier\"}], \"intent_ranking\": [{\"name\": \"makePayment\", \"confidence\": \"0.50\", \"utterance\": \"I would like to make payment\"}, {\"name\": \"checkCreditCardBalance\", \"confidence\": \"0.00\", \"utterance\": \"What is my credit card balance\"}, {\"name\": \"deny\", \"confidence\": \"0.00\", \"utterance\": \"not really\"}], \"text\": \"I want to pay my SBI card\"}"
```


License
----

MIT

   [url1]: <https://rasa.com/docs/rasa/tuning-your-model/>
   [url2]: <https://docs.docker.com/install/>
   [url3]: <https://rasa.com/docs/rasa/training-data-format>
   [url4]: <https://rasa.com/docs/rasa/components>
