# DA_225o_DeepLearning_Project_Team3
SLAM (Small Language Agentic Machine)

## Local setup for the python virtual env

```
python3 -m venv DL-PROJECT
source DL-PROJECT/bin/activate
pip install -r requirements.txt

```

## Running the project

### To run the streamlit app locally

```
cd src/UI
streamlit run main.py
```

### To build and run the UI on docker container

```

docker build -f src/deploy/container-ui/Dockerfile -t slam-ui-app .
docker run -p 8501:8501 slam-ui-app
OPEN APP  : http://localhost:8501

```

### To build and run the backend on docker container

```

docker build -f src/deploy/container-backend/Dockerfile -t slam-backend-app .
docker run -p 8000:8000 slam-backend-app
API request   : http://0.0.0.0:8000/ping

```

### To run the entire project containers using docker-compose

```
cd src/deploy
docker-compose up --build


### Run the Backend server
```
cd src/BACKEND
uvicorn main:app --reload
```