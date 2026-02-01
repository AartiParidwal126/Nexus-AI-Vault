# 🚀 Tesla Sales Intelligence: Multi-Agent AI System

A production-grade Agentic AI workflow that transforms raw sales data into actionable business strategies using Google Gemini LLM, PyTorch, and Scikit-Learn.

---

# Architecture: Multi-Agent Intelligence
This system follows a modular agentic workflow designed for scalability:

Agent 1 (Data Analyst):** Uses K-Means Clustering to identify patterns in Tesla's sales volume and pricing segments.
Agent 2 (Market Researcher):** Leverages Gemini 3.1 Flash to interpret data clusters and identify global market sentiment.
Agent 3 (Sales Strategist): Converts technical insights into 3 actionable "Revenue Growth" steps for the sales team.


# Tech Stack
 Category | Technology |
 Language** | Python 3.10+ |
 AI Model** | Google Gemini 3.1 Flash (Nano Banana model for high-fidelity reasoning) |
 Deep Learning** | PyTorch (High-performance Tensor manipulation) |
  Machine Learning| Scikit-Learn (K-Means, MinMaxScaler) |
  Data Processing | Pandas, Numpy |

---

# Quality Assurance (Google Flow)
Following the "Untested code is broken code" philosophy, this project is battle-tested:
 Automated Unit Tests:** Validates data integrity, clustering logic, and API integration.
 Latest Status:** `Ran 3 tests ... OK` 

---

# Future Roadmap & Scalability
The system is built with a **Pluggable Agent Architecture**, allowing us to add:
* **Agent 4 (Competitor Intelligence):** To compare Tesla's performance with competitors like BYD or Ford.
* **Agent 5 (Customer Sentiment):** To analyze real-time feedback from social media.
* **SaaS Vision:** Transforming this tool into a subscription-based platform where any business can upload CSVs for automated strategy generation.

---

# How to Run
1.  **Install Dependencies:**
    `pip install -r requirements.txt`
2.  **Set Environment Variables:**
    Create a `.env` file and add your `GOOGLE_API_KEY`.
3.  **Execute System:**
    `python app.py`
4.  **Run Tests:**
    `python test_agents.py`

---
Developed by Aarti - Building the future of SaaS with Agentic AI*