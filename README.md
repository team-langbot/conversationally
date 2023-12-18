# <img src="media/favicon.ico" alt="icon" width="32"/> Conversationally
Conversationally is a AI based conversational tutor created by a team of five students of Masters in Data Science course at UC Berkeley as part of the final Capstone project. It leverages multiple NLP techniques to create a conversation centric approach to second language learning.
## Project Artifacts
 Name|Type|Link
-|-|-
Deck|pdf| [capstone_presentation.pdf](media/capstone_presentation.pdf)
Promo Video|mp4| [conversationally_promo_small.mp4](media/conversationally_promo_small.mp4)
Demo Video|mov| [conversationally_demo.mov](media/conversationally_demo.mov)
Architecture Definition|pdf| [project_langbot_architecture.pdf](media/project_langbot_architecture.pdf)
Backend APIs|Code Repo|[project-langbot-webapp](https://github.com/team-langbot/project-langbot-webapp)
Web Application|Code Repo|[langbot-ui](https://github.com/team-langbot/langbot-ui)
Content Classification Model|Code Repo|[content-classification](https://github.com/team-langbot/content-classification)
Grammatical Error Correction Model|Code Repo|[model-gec](https://github.com/team-langbot/model_gec)
Response Generation|Code Repo|[GPT](https://github.com/team-langbot/GPT)
GEC - Intermediate Report|pdf|[GEC Experiments](BERT_based_Vanilla_Models-InitialExperiments_langbot_gec_plain_bert_experiments.pdf)

# Team

<div style="display:flex; align-items:center; justify-content: center;">
  <div style="float: left;width: 33.33%;">
    <img src="media/aastha.jpeg" alt="drawing" width="100px" style="border-radius: 100px"/>
    <div style="font-size:20">Aastha Khanna
    <a href="https://www.linkedin.com/in/aasthakh/"><img src="media/linkedin.png" width="32" style="background-color:white;border-radius: 16px"/></a>
    </div>
    <div>Aastha Khanna is a software development engineer at Amazon, currently working on Ring Virtual Security Guard security system.</div>
  </div>
  <div style="float: left;width: 33.33%;">
  <img src="media/isabel.jpeg" alt="drawing" width="100px" style="border-radius: 100px"/>

  <div style="font-size:20">Isabel Chan
  <a href="https://www.linkedin.com/in/waitingchan/"><img src="media/linkedin.png" width="32" style="background-color:white;border-radius: 16px"/></a>
  </div>
  <div>Isabel is a Data Engineer at , where she is involved in the ad measurement and support of various advertising products.</div>
  </div>
  <div style="float: left;width: 33.33%;">
<img src="media/jess.jpeg" alt="drawing" width="100px" style="border-radius: 100px"/>

<div style="font-size:20">Jess Matthews
<a href="https://www.linkedin.com/in/jessmatth/"><img src="media/linkedin.png" width="32" style="background-color:white;border-radius: 16px"/></a>
</div>
<div>Jess Matthews is VP of Global Product Management at Gartner where she oversees a portfolio of data-driven products supporting HR executives.</div>
  </div>
</div>

<div style="display:flex; align-items:center; justify-content: center;">
  <div style="width: 33.33%;">
<img src="media/mon.jpeg" alt="drawing" width="100px" style="border-radius: 100px"/>

<div style="font-size:20">Mon Young
<a href="https://www.linkedin.com/in/mon-young-a510901/"><img src="media/linkedin.png" width="32" style="background-color:white;border-radius: 16px"/></a>
</div>
<div>Mon Young founded ABiCO Capital Management, America branch. He presently directing the IT DevOps and Data Science divisions at Panasonic R&D Company of America.</div>
  </div>
  <div style="width: 33.33%;">
<img src="media/ram.jpeg" alt="drawing" width="100px" style="border-radius: 100px"/>

<div style="font-size:20">Ram Senthamarai
<a href="https://www.linkedin.com/in/ramsenth/"><img src="media/linkedin.png" width="32" style="background-color:white;border-radius: 16px"/></a>
</div>
<div>Ram is an experienced software engineer. He worked in the visual search domain in his last role at Amazon.</div>  </div>
</div>

# Approach
As part of this project, we implemented a chatbot for language learners to hold a scenario based conversation with. The chatbot is a web based application, built and hosted using AWS Amplify. It is powered by three models based on Natural Language Processing techniques. All our models are hosted on Amazon SageMaker and accessed through backend APIs.

<img src="media/hla.png" width="800" />

## Model 1:
The first model is focussed on keeping the conversation on topic, helping learner stay focussed on learning objective. We use a [Sentence Transformer](https://huggingface.co/sentence-transformers/multi-qa-MiniLM-L6-cos-v1) to generate the semantic embeddings for both Bot's output and user's response to it. A cosine similarity measure helps identify if user's response is on topic or not.

There are two key notebooks that have all the code related this model.

Notebook|Description
-|-
[content_classification.ipynb](https://github.com/team-langbot/content-classification/blob/main/content_classification.ipynb) | Experimentation with the Sentence Transformer model as well as the evaluation on test dataset.
[create-endpoint-01-sbert-Copy1.ipynb](https://github.com/team-langbot/content-classification/blob/main/create-endpoint-01-sbert-Copy1.ipynb) | Code for creating an Amazon SageMaker endpoint to serve inference calls.

## Model 2:
The second model is a token classification model that classifies each word in an input sentence as one of three classes - no error, gender mismatch error or number mismatch error. We have a separate model as explicit error identification keeps LLM based responses predictable and on-topic. It also makes language learning easier. We fine tuned [Beto](https://huggingface.co/dccuchile/bert-base-spanish-wwm-cased) which is Spanish BERT model on the [COWS-L2H](https://github.com/ucdaviscl/cowsl2h) dataset.

There are four key notebooks for GEC model:

Notebook|Description
-|-
[EDA.ipynb](https://github.com/team-langbot/model_gec/blob/main/EDA.ipynb)|Basic EDA on COWS-L2H dataset.
[gec_bert_data_prep.ipynb](https://github.com/team-langbot/model_gec/blob/main/gec_bert_data_prep.ipynb)|More GEC model specific EDA and Feature Engineering
[pytorch_gec_bert_plain.ipynb](https://github.com/team-langbot/model_gec/blob/main/pytorch_gec_bert_plain.ipynb)| Model training and evaluation
[gec_inference_setup.ipynb](https://github.com/team-langbot/model_gec/blob/main/deployment/gec_inference_setup.ipynb)| Creating SageMaker endpoint.

Finally, artifacts from each model training run are on [WandB](https://wandb.ai/langbot/projects) with project [langbot_gec_plain_top_performers](https://wandb.ai/langbot/langbot_gec_plain_top_performers) having the model that was finally deployed.

## Model 3:
The third model in our sequence is a generative model that handles generating hints about any error in user input. These hints are also called scaffolding. We experimented with multiple LLMs for this and finally selected [Mistral 7b](https://huggingface.co/docs/transformers/main/model_doc/mistral) model for generating scaffolding.

Key notebooks for this model:

Notebook|Description
-|-
[create-ep03-mistral2.ipynb](https://github.com/team-langbot/GPT/blob/main/create-ep03-mistral2.ipynb)|Code for creating SageMaker end point
[mistral2.ipynb](https://github.com/team-langbot/GPT/blob/main/mistral2.ipynb) | Experiments with Mistral model.
[langchain01.ipynb](https://github.com/team-langbot/GPT/blob/main/langchain01.ipynb) | Experiments with Langchain and agents.

# Backend API
We created an AWS Lambda that acts as the backend for inference calls on each of the three models and is exposed as a single endpoint via API Gateway. [Backend Code Repository](https://github.com/team-langbot/project-langbot-webapp) holds the code for this. In particular, [index.py](https://github.com/team-langbot/project-langbot-webapp/blob/main/amplify/backend/function/projectlangbotapi/src/index.py) stores the code that orchestrates the model calls and parses the individual responses into the API response shown on the frontend.

# Web Application
The code for our amplify based webapp is at [Webapp Code Repository](https://github.com/team-langbot/langbot-ui) with [index.js](https://github.com/team-langbot/langbot-ui/blob/main/src/index.js) as the entry point for the app.

## References
Below are the references we used during our project.

Northern Illinois University Center for Innovative Teaching and Learning. (2012). [Instructional scaffolding. In Instructional guide for university faculty and teaching assistants.](https://www.niu.edu/citl/resources/guides/instructional-guide/instructional-scaffolding-to-improve-learning.shtml) Retrieved from https://www.niu.edu/citl/resources/guides/instructional-guide

[Long-Term Language Retention for Students of a Second Language: A Review of the Literature](https://repository.stcloudstate.edu/cgi/viewcontent.cgi?article=1039&context=ed_etds)

Davidson, S., Yamada, A., Mira, P.F., Carando, A., Gutierrez, C.H., & Sagae, K. (2020). [Developing NLP Tools with a New Corpus of Learner Spanish](https://aclanthology.org/2020.lrec-1.894/). International Conference on Language Resources and Evaluation.

Kiros, R., Zhu, Y., Salakhutdinov, R., Zemel, R.S., Urtasun, R., Torralba, A., & Fidler, S. (2015). [Skip-Thought Vectors. Neural Information Processing Systems.](https://www.semanticscholar.org/paper/Skip-Thought-Vectors-Kiros-Zhu/6e795c6e9916174ae12349f5dc3f516570c17ce8)

Alexis Conneau, Douwe Kiela, Holger Schwenk, Loïc Barrault, and Antoine Bordes. 2017. [Supervised Learning of Universal Sentence Representations from Natural Language Inference Data](https://aclanthology.org/D17-1070/). In Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing, pages 670–680, Copenhagen, Denmark. Association for Computational Linguistics.

Daniel Cer, Yinfei Yang, Sheng-yi Kong, Nan Hua, Nicole Limtiaco, Rhomni St. John, Noah Constant, Mario Guajardo-Cespedes, Steve Yuan, Chris Tar, Brian Strope, and Ray Kurzweil. 2018. [Universal Sentence Encoder for English](https://aclanthology.org/D18-2029/). In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, pages 169–174, Brussels, Belgium. Association for Computational Linguistics.

Nils Reimers and Iryna Gurevych. 2019. [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://aclanthology.org/D19-1410/). In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 3982–3992, Hong Kong, China. Association for Computational Linguistics.

Sun, Xin, Tao Ge, Shuming Ma, Jingjing Li, Furu Wei and Houfeng Wang. “[A Unified Strategy for Multilingual Grammatical Error Correction with Pre-trained Cross-Lingual Language Model](https://www.semanticscholar.org/paper/A-Unified-Strategy-for-Multilingual-Grammatical-Sun-Ge/05b90590b1ef911703d83399ea1ff5f01faa44d5).” International Joint Conference on Artificial Intelligence (2022).

Linting Xue, Noah Constant, Adam Roberts, Mihir Kale, Rami Al-Rfou, Aditya Siddhant, Aditya Barua, and Colin Raffel. 2021. [mT5: A Massively Multilingual Pre-trained Text-to-Text Transformer](https://aclanthology.org/2021.naacl-main.41/). In Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 483–498, Online. Association for Computational Linguistics.

Yuejiao Fei, Leyang Cui, Sen Yang, Wai Lam, Zhenzhong Lan, and Shuming Shi. 2023. [Enhancing Grammatical Error Correction Systems with Explanations](https://aclanthology.org/2023.acl-long.413). In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 7489–7501, Toronto, Canada. Association for Computational Linguistics.

Muntsa Padró, Miguel Ballesteros, Héctor Martínez, and Bernd Bohnet. 2013. [Finding Dependency Parsing Limits over a Large Spanish Corpus](https://aclanthology.org/I13-1123/). In Proceedings of the Sixth International Joint Conference on Natural Language Processing, pages 942–946, Nagoya, Japan. Asian Federation of Natural Language Processing.

Cañete, José Luis González, Gabriel Chaperon, Rodrigo Fuentes, Jou-Hui Ho, Hojin Kang and Jorge P'erez. "[Spanish Pre-trained BERT Model and Evaluation Data](https://www.semanticscholar.org/paper/Spanish-Pre-trained-BERT-Model-and-Evaluation-Data-Ca%C3%B1ete-Chaperon/79926aa63d4daee6af06a8e9a7c2480b31cb7ed9)." ArXiv abs/2308.02976 (2023): n. pag.

[Spanish Grammatical Error Correction](https://diligent-raver-536.notion.site/Spanish-Grammatical-Error-Correction-88d0f0d1d090412baf4c52cdf87a0468)

[A Simple Named Entity Recognition Model using BERT and Keras](https://github.com/datasci-w266/2021-fall-main/blob/f6387a405f307ddc576c363b8b7885869fe224d0/materials/Bert/BERT_T5_NER_2_3_030521.ipynb)

[Lesson notebook 8 - Parsing](https://colab.research.google.com/github/datasci-w266/2023-fall-main/blob/master/materials/lesson_notebooks/lesson_8_Parsing.ipynb)
