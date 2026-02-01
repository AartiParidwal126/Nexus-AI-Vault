import os
from urllib import response
import pandas as pd 
from dotenv import load_dotenv


load_dotenv()

api_key = os.getenv("GOOGLE_API-KEY")

df = pd.read_csv("tesla_data.csv")

print(df.head())



numeric_df = df[['Open', 'High', 'Low', 'Close', 'Volume']]

# 4. Check Result
print("Data with only numbers:")
print(numeric_df.head(2))



from sklearn.cluster import KMeans
import torch 
data_tensor = torch.tensor(numeric_df.values)
print("pytorch tensor ready hai ")

print(data_tensor[:2])  


from sklearn.preprocessing import MinMaxScaler
 #scalling 
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(numeric_df)

#tensor update

tensor_data = torch.tensor(scaled_data, dtype=torch.float32)
print("data_tensor[:2]")




import google.generativeai as genai

genai.configure(api_key="AIzaSyDSmdohmysgrfgXLyTs1ngjabGyZMfR_AM")

for m in genai.list_models():
    if 'generateContent' in m.supported_generation_methods:
        print(m.name)


        import google.generativeai as genai
import time

# 1. Setup
genai.configure(api_key="AIzaSyDe44MBBgmBl8j-lN_VL1L9zdMv-OqDgR0")
model = genai.GenerativeModel('gemini-3-flash-preview')

def analyze_data(prompt):
    try:
        # 2. API Call (Model se jawab mangna)
        response = model.generate_content(prompt)
        
        # Check karna ki AI ne jawab diya ya nahi
        if response.candidates and response.candidates[0].content.parts:
            return response.candidates[0].content.parts[0].text
        else:
            return "AI did not give an answer (Response blocked)."

    except Exception as e:
        # 3. Rate Limit (429) Handling
        if "429" in str(e):
            print("Quota full, 20s wait...")
            time.sleep(20)
            # Dobara koshish karna
            new_response = model.generate_content(prompt)
            return new_response.text
        else:
            return f"Error: {e}"


# --- STEP 1: AGENT 1 (DATA ANALYST) ---
def run_agent_1(): # <-- यहाँ अंत में ':' होना ज़रूरी है
    print("\n🚀 Agent 1 (Data Analyst) is working...")
    
    # डेटा लोड करना
    df = pd.read_csv('tesla_data.csv')
    
    # डेटा को क्लस्टर करना
    data_points = df[['Close', 'Volume']].fillna(0)
    
    kmeans = KMeans(n_clusters=3, random_state=42)
    df['Cluster'] = kmeans.fit_predict(data_points)
    
    counts = df['Cluster'].value_counts().to_dict()
    summary = f"Tesla data analysis complete. We found 3 clusters: {counts}."
    
    print("✅ Agent 1: Clustering and Analysis Finished.")
    return summary

# execute Agent 1
final_summary = run_agent_1()
print("final summaery from agent 1")



# Agent 2 markaet reseacher

def run_agent_2(summary_from_agent1):
    print("\n🔍 Agent 2 (Market Researcher) is investigating...")
    
    # Gemini के लिए सवाल (Prompt)
    prompt = f"""
    Context: Agent 1 found these data clusters: {summary_from_agent1}.
    Task: Explain what these clusters mean for Tesla's business and if any market trends are visible.
    Provide a professional research insight.
    """
    
    # Gemini से जवाब मांगना
    insight = analyze_data(prompt)
    
    print("✅ Agent 2: Research Insight Generated.")
    return insight

# --- EXECUTION ---
# Agent 1 के रिजल्ट को Agent 2 में भेजना
market_report = run_agent_2(final_summary)

print("\n" + "="*50)
print("📑 STRATEGIC RESEARCH REPORT")
print("="*50)
print(market_report)


# --- STEP 3: AGENT 3 (SALES STRATEGIST) ---
def run_agent_3(research_from_agent2):
    print("\n📝 Agent 3 (Sales Strategist) is creating action plan...")
    
    # 1. AI को काम समझाना
    prompt = f"""
    Context: Use this research report: {research_from_agent2}.
    Task: Provide 3 practical 'Action Steps' for the sales team to increase revenue.
    Note: Keep it simple, bulleted and actionable.
    """
    
    # 2. Gemini से प्लान बनवाना
    action_plan = analyze_data(prompt)
    
    print("✅ Agent 3: Action Plan Ready.")
    return action_plan

# --- FINAL EXECUTION ---
# Agent 3 को Agent 2 का रिजल्ट देना
final_plan = run_agent_3(market_report)

# स्क्रीन पर फाइनल आउटपुट दिखाना
print("\n" + "*"*50)
print("🚀 FINAL SALES ACTION PLAN FOR TESLA")
print("*"*50)
print(final_plan)