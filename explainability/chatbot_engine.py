import json
import urllib.request
import urllib.error

def get_local_response(query):
    query_lower = query.lower()
    
    # Tabular models query
    if any(k in query_lower for k in ["xgboost", "random forest", "svm", "logistic regression", "tabular", "clinical risk", "machine learning", "ml model"]):
        return """
<div style='background-color: rgba(49, 130, 206, 0.1); border-left: 4px solid #3182CE; padding: 15px; border-radius: 8px; margin-bottom: 10px;'>
    <h4 style='color: #3182CE; margin-top: 0; margin-bottom: 8px; font-family: "Outfit", sans-serif;'>📋 Clinical Risk Factors Machine Learning Ensemble</h4>
    <p style='margin: 0; font-size: 14px; line-height: 1.5; color: #E2E8F0;'>
        Pulse AI integrates four state-of-the-art machine learning models trained on clinical risk features (age, chest pain type, max heart rate, blood pressure, ST-segment depression):
    </p>
    <ul style='margin: 8px 0 0 0; padding-left: 20px; font-size: 13px; color: #A0AEC0;'>
        <li><strong>XGBoost (Best Model)</strong>: Test Accuracy of <strong>89.1%</strong> (Cross-Validation Mean: 82.6%). Extremely robust against clinical outliers.</li>
        <li><strong>Random Forest</strong>: Test Accuracy of <strong>88.6%</strong> (CV Mean: 83.8%). Offers high ensemble stability.</li>
        <li><strong>SVM (Support Vector Machine)</strong>: Test Accuracy of <strong>88.6%</strong> (CV Mean: 83.5%). Excellent for separation boundaries.</li>
        <li><strong>Logistic Regression</strong>: Test Accuracy of <strong>88.0%</strong> (CV Mean: 84.1%). Highly interpretable baseline model.</li>
    </ul>
</div>
"""

    # SHAP query
    if any(k in query_lower for k in ["shap", "waterfall", "contribution", "tabular explanation", "feature importance"]):
        return """
<div style='background-color: rgba(159, 122, 234, 0.1); border-left: 4px solid #9F7AEA; padding: 15px; border-radius: 8px; margin-bottom: 10px;'>
    <h4 style='color: #9F7AEA; margin-top: 0; margin-bottom: 8px; font-family: "Outfit", sans-serif;'>🧭 SHAP (SHapley Additive exPlanations)</h4>
    <p style='margin: 0; font-size: 14px; line-height: 1.5; color: #E2E8F0;'>
        <strong>SHAP</strong> is a game-theoretic approach used to explain the output of machine learning models. It calculates the marginal contribution of each patient feature to the final diagnostic probability:
    </p>
    <ul style='margin: 8px 0 0 0; padding-left: 20px; font-size: 13px; color: #A0AEC0;'>
        <li><strong>Positive SHAP values (Red)</strong> push the prediction towards high cardiac risk (e.g. Asymptomatic Chest Pain, high exercise-induced ST depression).</li>
        <li><strong>Negative SHAP values (Blue)</strong> push the prediction towards low cardiac risk (e.g. Normal resting ECG, Up-sloping ST segment).</li>
        <li>This allows clinicians to see exactly <em>why</em> a patient was classified as high or low risk.</li>
    </ul>
</div>
"""

    # Echo DL model query
    if any(k in query_lower for k in ["echo", "resnet", "video", "deep learning", "dl model", "apical", "ejection fraction"]):
        return """
<div style='background-color: rgba(49, 151, 149, 0.1); border-left: 4px solid #319795; padding: 15px; border-radius: 8px; margin-bottom: 10px;'>
    <h4 style='color: #319795; margin-top: 0; margin-bottom: 8px; font-family: "Outfit", sans-serif;'>🫀 Echocardiography Deep Learning Module</h4>
    <p style='margin: 0; font-size: 14px; line-height: 1.5; color: #E2E8F0;'>
        The Echo analysis runs a spatiotemporal deep learning model based on the **EchoNet-Dynamic 3D-ResNet18** architecture:
    </p>
    <ul style='margin: 8px 0 0 0; padding-left: 20px; font-size: 13px; color: #A0AEC0;'>
        <li><strong>Target Modality</strong>: Apical 4-chamber (A4C) echocardiography videos (.mp4/.avi).</li>
        <li><strong>Performance Metrics</strong>: Area Under ROC is <strong>0.91</strong>. Mean local inference speed is <strong>&lt; 2.4s</strong> on CPU.</li>
        <li><strong>Task</strong>: Automatically detects markers of reduced ejection fraction (EF &lt; 50%) and myocardial wall motion abnormalities.</li>
    </ul>
</div>
"""

    # Grad-CAM query
    if any(k in query_lower for k in ["grad-cam", "gradcam", "heatmap", "saliency", "visual explanation", "attention"]):
        return """
<div style='background-color: rgba(221, 107, 32, 0.1); border-left: 4px solid #DD6B20; padding: 15px; border-radius: 8px; margin-bottom: 10px;'>
    <h4 style='color: #DD6B20; margin-top: 0; margin-bottom: 8px; font-family: "Outfit", sans-serif;'>🔥 Grad-CAM (Gradient-weighted Class Activation Mapping)</h4>
    <p style='margin: 0; font-size: 14px; line-height: 1.5; color: #E2E8F0;'>
        <strong>Grad-CAM</strong> is a visualization technique for deep learning networks. It uses the gradients of the target class (abnormal cardiac function) flowing into the final 3D-Convolutional layer to produce coarse localization heatmaps:
    </p>
    <ul style='margin: 8px 0 0 0; padding-left: 20px; font-size: 13px; color: #A0AEC0;'>
        <li>Highlights the spatial regions of the apical 4-chamber video frames that were critical to the classification.</li>
        <li><strong>Red/Warm regions</strong> highlight focal areas of interest, such as abnormal movement in the left ventricle wall or blood flow dynamics in the chambers.</li>
        <li>Ensures deep neural network decisions are transparent and auditable for clinical audits.</li>
    </ul>
</div>
"""

    # Late Fusion query
    if any(k in query_lower for k in ["late fusion", "fusion", "multimodal", "combine", "ensemble", "weight"]):
        return """
<div style='background-color: rgba(49, 130, 206, 0.1); border-left: 4px solid #3182CE; padding: 15px; border-radius: 8px; margin-bottom: 10px;'>
    <h4 style='color: #3182CE; margin-top: 0; margin-bottom: 8px; font-family: "Outfit", sans-serif;'>🔗 Late Fusion Multimodal Integration</h4>
    <p style='margin: 0; font-size: 14px; line-height: 1.5; color: #E2E8F0;'>
        Pulse AI implements a **probability-level late fusion** approach to unify structured tabular measurements and unstructured video content:
    </p>
    <ul style='margin: 8px 0 0 0; padding-left: 20px; font-size: 13px; color: #A0AEC0;'>
        <li><strong>Clinical Weights</strong>: <strong>60% weight</strong> is assigned to the Echocardiography model (capturing real-time wall motion & EF) and <strong>40% weight</strong> to the Clinical Risk Factors ensemble (capturing demographic & history statistics).</li>
        <li><strong>Output Risk Classes</strong>: Low Risk (probability &le; 0.40), Moderate Risk (0.40 &lt; probability &le; 0.60), and High Risk (probability &gt; 0.60).</li>
        <li>Late fusion outperforms single modalities by providing a cross-referenced diagnosis.</li>
    </ul>
</div>
"""

    # ECG / ST Slope / Oldpeak query
    if any(k in query_lower for k in ["ecg", "st slope", "oldpeak", "electrocardiogram", "depression", "angina"]):
        return """
<div style='background-color: rgba(229, 62, 62, 0.1); border-left: 4px solid #E53E3E; padding: 15px; border-radius: 8px; margin-bottom: 10px;'>
    <h4 style='color: #E53E3E; margin-top: 0; margin-bottom: 8px; font-family: "Outfit", sans-serif;'>⚡ ECG & Exercise Stress Indicators</h4>
    <p style='margin: 0; font-size: 14px; line-height: 1.5; color: #E2E8F0;'>
        The ECG and exercise stress test parameters entered in the console are strong clinical markers for coronary heart disease:
    </p>
    <ul style='margin: 8px 0 0 0; padding-left: 20px; font-size: 13px; color: #A0AEC0;'>
        <li><strong>Oldpeak</strong>: Measures ST segment depression induced by exercise relative to rest. Values above 1.5 suggest myocardial ischemia.</li>
        <li><strong>ST Slope</strong>: The slope of the peak exercise ST segment. An up-sloping segment is normal, while a flat or down-sloping segment indicates significant coronary stenosis.</li>
        <li><strong>Exercise Angina</strong>: Indicates whether the patient experienced chest pain during the stress test, indicating oxygen-supply mismatches.</li>
    </ul>
</div>
"""

    # General clinical risk advice / Diet / Prevention query
    if any(k in query_lower for k in ["diet", "advice", "exercise", "prevention", "cardio", "healthy", "bp", "blood pressure", "cholesterol"]):
        return """
<div style='background-color: rgba(72, 187, 120, 0.1); border-left: 4px solid #48BB78; padding: 15px; border-radius: 8px; margin-bottom: 10px;'>
    <h4 style='color: #48BB78; margin-top: 0; margin-bottom: 8px; font-family: "Outfit", sans-serif;'>🥗 Cardiovascular Health Guidelines</h4>
    <p style='margin: 0; font-size: 14px; line-height: 1.5; color: #E2E8F0;'>
        Based on the American Heart Association (AHA) and ACC clinical practice guidelines:
    </p>
    <ul style='margin: 8px 0 0 0; padding-left: 20px; font-size: 13px; color: #A0AEC0;'>
        <li><strong>Nutritional Management</strong>: Follow a DASH or Mediterranean-style diet. Restrict sodium to &lt; 2,000 mg per day and minimize trans fats.</li>
        <li><strong>Exercise Routine</strong>: Aim for at least 150 minutes of moderate-intensity aerobic exercise (like brisk walking) or 75 minutes of vigorous exercise weekly.</li>
        <li><strong>Lipid Control</strong>: Maintain LDL cholesterol &lt; 100 mg/dL (or &lt; 70 mg/dL for patients at high clinical risk).</li>
        <li><strong>Blood Pressure Control</strong>: Manage blood pressure to target &lt; 130/80 mmHg using lifestyle modifications and appropriate therapeutics.</li>
    </ul>
</div>
"""
    return None

def get_chatbot_response(query, api_key=None, chat_history=[]):
    # 1. If an API key is provided, attempt live generative response
    if api_key and api_key.strip():
        # Clean up any potential HTML markers if generating through API, we want raw text/markdown
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.1-flash-lite:generateContent?key={api_key.strip()}"
        
        system_instruction = (
            "You are Pulse AI, a clinical chat assistant companion for a Multimodal Heart Disease Diagnostic System. "
            "Your goal is to answer questions accurately, clearly, and helpfully. "
            "The project integrates Clinical Tabular Risk Factors (using a performance-weighted XGBoost & SVM ensemble with 89.1% test accuracy and 82.6% cross-validation accuracy) "
            "and Echocardiography Video Analysis (using an EchoNet-Dynamic 3D-ResNet18 model with 0.91 AUC and <2.4s latency, explained via Grad-CAM heatmaps). "
            "The modalities are combined using clinically weighted Late Fusion (60% weight on Echo, 40% weight on tabular). "
            "Keep your answers concise, structured, and clinically relevant. If the user asks simple, general, everyday, or non-medical questions, "
            "answer them in a friendly, concise manner, but relate them to cardiology or health when relevant."
        )
        
        contents = []
        for chat in chat_history[-6:]:
            contents.append({
                "role": "user" if chat.get("user") else "model",
                "parts": [{"text": chat["user"] if chat.get("user") else chat["bot"]}]
            })
            
        contents.append({
            "role": "user",
            "parts": [{"text": query}]
        })
        
        payload = {
            "contents": contents,
            "systemInstruction": {
                "parts": [{"text": system_instruction}]
            },
            "generationConfig": {
                "maxOutputTokens": 600,
                "temperature": 0.7
            }
        }
        
        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=8) as response:
                res_data = json.loads(response.read().decode("utf-8"))
                text_response = res_data["candidates"][0]["content"]["parts"][0]["text"]
                return text_response
        except urllib.error.HTTPError as e:
            error_msg = e.read().decode("utf-8")
            try:
                error_json = json.loads(error_msg)
                return f"**Gemini API Error:** {error_json['error']['message']}"
            except Exception:
                return f"**HTTP Error:** {e.code} - {e.reason}. Please check if your API Key is valid."
        except Exception as e:
            return f"**Connection Error:** Could not contact Gemini API servers. {e}"

    # 2. If no API key, fall back to local clinical knowledge engine
    local_resp = get_local_response(query)
    if local_resp:
        return local_resp
        
    # Local fallback for general questions when no API key is set
    return """
<div style='background-color: rgba(113, 128, 150, 0.1); border-left: 4px solid #718096; padding: 15px; border-radius: 8px; margin-bottom: 10px;'>
    <h4 style='color: #E2E8F0; margin-top: 0; margin-bottom: 8px; font-family: "Outfit", sans-serif;'>🤖 Pulse AI Clinical Assistant (Offline Fallback)</h4>
    <p style='margin: 0; font-size: 14px; line-height: 1.6; color: #CBD5E0;'>
        I am currently operating in <strong>offline keyword-matching mode</strong>. To enable me to answer general, everyday, or custom medical questions:
    </p>
    <div style='background: rgba(255, 255, 255, 0.05); padding: 10px; border-radius: 6px; margin: 10px 0; font-family: monospace; font-size: 12px; color: #3182CE;'>
        Please input a Gemini API Key in the sidebar input field!
    </div>
    <p style='margin: 0; font-size: 13px; color: #A0AEC0;'>
        Otherwise, you can ask me about predefined clinical modules:
    </p>
    <ul style='margin: 4px 0 0 0; padding-left: 20px; font-size: 13px; color: #A0AEC0;'>
        <li>"What is the accuracy of the XGBoost model?"</li>
        <li>"Explain the EchoNet ResNet model."</li>
        <li>"What is Grad-CAM?"</li>
        <li>"How does the SHAP explainability work?"</li>
        <li>"Give me advice on diet and blood pressure."</li>
    </ul>
</div>
"""
