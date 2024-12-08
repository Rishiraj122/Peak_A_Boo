# Image Segmentation Web App

A web application that removes image backgrounds using YOLOv8 segmentation model.

## Setup Instructions

### Local Development
1. Clone the repository:
```bash
git clone https://github.com/Rishiraj122/Peak_A_Boo.git
cd Peak_A_Boo


python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate

pip install -r requirements.txt

wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m-seg.pt

# Run backend
python app.py

# In another terminal, run frontend
cd React_web/image-segmentation
npm install
npm start