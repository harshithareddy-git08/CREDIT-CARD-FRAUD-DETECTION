🛡️CREDIT-CARD-FRAUD-DETECTION

  📌 Project Overview
  
This project implements a **machine learning pipeline** to detect fraudulent credit card transactions using the anonymized dataset (`creditcard.csv`). Fraud detection is a critical challenge in financial systems due to the rarity of fraudulent cases, the need for real-time detection, and the importance of scalable deployment.

 ✨ Features
 
- **Data Preprocessing**: Cleans and normalizes transaction data.  
- **Imbalanced Data Handling**: Uses undersampling/SMOTE to address fraud rarity.  
- **Machine Learning Models**: Logistic Regression, Random Forest, Gradient Boosting.  
- **Evaluation Metrics**: Precision, Recall, F1-score, ROC-AUC.  
- **Model Deployment**: REST API endpoints with FastAPI.  
- **Scalable API**: Uvicorn server for lightweight deployment.  
- **Web Interface**: Jinja2 templates for user-friendly interaction.  
- **Model Serialization**: Save and load models with `joblib`.  

 
 ⚙️ Tech Stack
 
- **pandas** → Data preprocessing  
- **numpy** → Numerical computations  
- **scikit-learn** → ML models & evaluation  
- **joblib** → Model persistence  
- **fastapi** → REST API framework  
- **uvicorn** → ASGI server  
- **python-multipart** → File uploads  
- **jinja2** → Web templating  


🔍 Workflow

1. **Data Preprocessing**
   - Load and clean dataset (`creditcard.csv`).
   - Handle imbalance with SMOTE/undersampling.
   - Normalize features.

2. **Model Training**
   - Train classification models.
   - Evaluate with Precision, Recall, F1, ROC-AUC.

3. **Model Deployment**
   - Save trained models with `joblib`.
   - Expose prediction endpoints via FastAPI.
   - Run API with Uvicorn.

4. **Usage**
   - Send transaction data to API.
   - Receive fraud probability and classification result.

     

 🚀 How to Run
 
1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/creditcard-fraud-detection.git
   cd creditcard-fraud-detection
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the model**
   ```bash
   python train_model.py
   ```

4. **Start the API**
   ```bash
   uvicorn main:app --reload
   ```

5. **Test the API**
   ```bash
   curl -X POST "http://127.0.0.1:8000/predict" \
   -H "Content-Type: application/json" \
   -d '{"amount":200.0,"time":12345,"V1":-1.2,...}'

🎓 Learning Outcomes

By completing this project, you will:
- Understand **imbalanced dataset challenges**.  
- Gain experience with **data preprocessing and feature scaling**.  
- Apply **classification algorithms** for fraud detection.  
- Explore **evaluation metrics beyond accuracy**.  
- Build and deploy a **REST API with FastAPI**.  
- Practice **model serialization** for production.  
- Learn **scalable deployment** using Uvicorn.  


📈 Future Enhancements

- Integration with **deep learning models** (e.g., LSTMs).  
- Real-time **streaming fraud detection** with Apache Kafka.  
- Interactive **dashboard** for monitoring fraud detection performance.  


📊 System Architecture (Pipeline)

Dataset (creditcard.csv) → Preprocessing → Model Training → Evaluation → Joblib Model → FastAPI → Uvicorn → User/API Client

👩‍💻 Author

Harshitha C

🔗GitHub: https://github.com/harshithareddy-8 
🔗Linkedin: https://www.linkedin.com/in/harshitha-c-7059412a9
