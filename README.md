# Active Learning demo on Openshift.

This is active learning demo using Label Studio and LabelStudio ML backend. This demo trains a model for vegetable classfication and also model is actively trained from label studio.

## References

  ### Training Data From Kaggle: 
  
      https://www.kaggle.com/datasets/misrakahmed/vegetable-image-dataset/download?datasetVersionNumber=1

  ### Model Reference
  
      https://www.kaggle.com/code/theeyeschico/vegetable-classification-using-transfer-learning

## Model Training
  
   Download Training Data from Kaggle references above and extract the data under model-training/Vegetable Images
   
   ### Training the model locally
   
        cd activelearning-demo/model-training
        python3 -m venv env
        source env/bin/activate
        pip install -r ./requirements.txt
        python3 model_training.py

   ### Training the model with RHODS
   
       Use RHODS Project with MinIO Server and establish the data connection and launch Notebook.
          
          Open Notebook : model_training.ipynb

## Active Learning Label Studio Backend

   ### Local Active Learning LabelStudioML backend server :
   
        python3 -m venv env
        source env/bin/activate
        pip install -r ./requirements.txt
        label-studio-ml start active-learning-labelstudio-ml-backend
    
