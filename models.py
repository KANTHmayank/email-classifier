import os
import pandas as pd
import openai
import streamlit as st

openai.api_key = st.secrets["OPENAI_API_KEY"]

# Prepare few-shot examples
DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "combined_emails_with_natural_pii.xlsx")
df = pd.read_excel(DATA_PATH).rename(columns={"email": "email_body", "type": "category"})

# pick up to 3 samples per category
few_shot = (
    df.groupby("category")
      .apply(lambda g: g.sample(n=min(len(g), 3), random_state=42))
      .reset_index(drop=True)
)
few_shot_block = "\n\n".join(
    f"Email: {row['email_body']}\nCategory: {row['category']}"
    for _, row in few_shot.iterrows()
)

def classify_email(email_text: str) -> str:
    prompt = (
         "You are a support-ticket classifier. Possible categories:\n"
         "  • Billing Issues\n"
         "  • Technical Support\n"
         "  • Account Management\n"
         "  • Others\n\n"
         "Here are some examples:\n\n"
         f"{few_shot_block}\n\n"
         "Now classify this new email.\n\n"
         f"Email: {email_text}\n"
         "Category:"
     )
    resp = openai.completions.create(
        model="gpt-3.5-turbo-instruct",
        prompt=prompt,
        max_tokens=12,
        temperature=0.0,
        stop=["\n"]
    )
    return resp.choices[0].text.strip()
