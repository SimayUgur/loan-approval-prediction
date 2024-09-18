FROM python:3.8.19


# Çalışma dizinini oluştur
WORKDIR /app

# Uygulama dosyalarını kopyala (öncelikle model ve scaler dosyaları dahil olmak üzere her şeyi)
COPY . /app

COPY loan_approval_model.pkl /app/
COPY rb_scaler.pkl /app/


# Gerekli bağımlılıkları yükle
RUN pip install -r requirements.txt

# Flask uygulamasını başlat
CMD ["python", "app.py"]
