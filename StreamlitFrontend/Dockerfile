FROM python:3.10

COPY requirements.txt streamlit/requirements.txt

WORKDIR /streamlit

RUN pip install -r requirements.txt

COPY . /streamlit

EXPOSE 8501

CMD ["streamlit", "run", "app.py"]