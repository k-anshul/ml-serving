# ml-serving

https://docs.google.com/document/d/1hZ-Fz-DzbU1irM80Ni_6Ilt2neDP8Z2hdmdlnuafJHg/edit?usp=sharing

## locally-test-steps
1. run mongo and service using docker-compose 
    - cmd : docker compose up -d
2. Import postman collection from resources folder 
3. Change file path to local path
4. Or use swagger UI to test the APIs
    - link : http://localhost:8000/docs


# Steps for training model
    - create model metadata using http://localhost:8000/docs#/Models/train_model_app_train_post
    - use model id and version id from the response above and upload bad image using http://localhost:8000/docs#/Models/train_model_app_train__model_id___version___label__put
    - similary upload good image
    - submit model for training using http://localhost:8000/docs#/Models/train_model_app_train_submit__model_id___version__post
