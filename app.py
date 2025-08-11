import streamlit as st
import ast
from transformers import TFAutoModelForSeq2SeqLM, AutoTokenizer, pipeline
import google.generativeai as genai

# Configure Gemini API key
genai.configure(api_key="AIzaSyBnId5QMwl5XaQ0qiVp_re5AARjQ5PjMqY")
gemini_model = genai.GenerativeModel('models/gemini-2.5-flash')

# Cache the summarizer model
@st.cache_resource
def load_summarizer_model():
    model = TFAutoModelForSeq2SeqLM.from_pretrained("suryaummadi/Flan-T5-Short-review-summarizer")
    tokenizer = AutoTokenizer.from_pretrained("suryaummadi/Flan-T5-Short-review-summarizer")
    return model, tokenizer

model, tokenizer = load_summarizer_model()

# Load sentiment pipeline once and cache it
@st.cache_resource
def load_sentiment_pipeline():
    return pipeline(
        "text-classification",
        model="suryaummadi/review-roberta-customer-experience-analytics",
        return_all_scores=True,
    )

sentiment_pipe = load_sentiment_pipeline()

# Summarization function
def summarize_review(review):
    input_text = "summarize: " + review
    inputs = tokenizer(
        input_text,
        return_tensors="tf",
        padding=True,
        truncation=True,
        max_length=512,
    )
    summary_ids = model.generate(
        **inputs,
        max_new_tokens=20,
        min_length=2,
        num_beams=4,
        length_penalty=0.8,
        early_stopping=True,
    )
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Extract pros and cons using Gemini
def extract_pros_cons_from_review(review_text):
    prompt = f"""
Extract pros and cons from the following review. Return ONLY a valid Python dictionary without any markdown or code formatting.

Review: "{review_text}"
"""
    response = gemini_model.generate_content(prompt)
    output_text = response.text.strip()

    try:
        output_dict = ast.literal_eval(output_text)
    except Exception as e:
        raise ValueError(f"Failed to parse model output. Output was:\n{output_text}\nError: {e}")

    return output_dict

# Sentiment analysis function
def get_top_sentiment(text):
    outputs = sentiment_pipe(text)
    scores = outputs[0]
    top = max(scores, key=lambda x: x['score'])
    label = top['label']  # Already 'negative', 'neutral', or 'positive'
    percentage = top['score'] * 100
    return f"The customer feels {percentage:.1f}% {label} about the product."

# Streamlit UI
st.set_page_config(page_title="📝 AI Product Review Summarizer & Pros/Cons Extractor", layout="wide")
st.title("📝 AI Product Review Summarizer & Pros/Cons Extractor")
st.write("Enter up to 5 product reviews. Click 'Analyze' to get summaries, extracted pros and cons, and sentiment analysis.")

review_inputs = []
cols = st.columns(5)
for i in range(5):
    with cols[i]:
        review = st.text_area(f"Review {i+1}", height=150)
        review_inputs.append(review.strip())

if st.button("Analyze Reviews"):
    st.subheader("🔍 Analysis Results")

    for i, review in enumerate(review_inputs):
        if review:
            with st.expander(f"Review {i+1} Results", expanded=True):
                col1, col2, col3 = st.columns([3, 3, 2])  # 3 columns

                with col1:
                    st.markdown(f"**Original Review {i+1}:**")
                    st.write(review)

                with col2:
                    # Summarize
                    summary = summarize_review(review)
                    st.markdown("**Summarized Review:**")
                    st.success(summary)

                    # Pros/Cons
                    try:
                        pros_cons = extract_pros_cons_from_review(review)
                        st.markdown("**Pros:**")
                        if pros_cons.get("pros"):
                            for item in pros_cons["pros"]:
                                st.markdown(f"- {item}")
                        else:
                            st.info("No pros found.")

                        st.markdown("**Cons:**")
                        if pros_cons.get("cons"):
                            for item in pros_cons["cons"]:
                                st.markdown(f"- {item}")
                        else:
                            st.info("No cons found.")
                    except Exception as e:
                        st.warning(f"Failed to extract pros and cons: {e}")

                with col3:
                    # Sentiment
                    sentiment_text = get_top_sentiment(review)
                    st.markdown("**Customer Experience Analytics:**")
                    st.info(sentiment_text)
