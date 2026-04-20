FROM python:3.13-slim

COPY requirements_app.txt .
RUN pip3 install --no-cache-dir --root-user-action=ignore -r requirements_app.txt

# Copy the app
COPY model_pth .

# Custom function
COPY model.py .

COPY app.py .

RUN useradd -m appuser
USER appuser

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501"]
