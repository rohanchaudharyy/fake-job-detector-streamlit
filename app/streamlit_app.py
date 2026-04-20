import streamlit as st
from dotenv import load_dotenv

from src.inference.predict_and_explain import HybridDetector


load_dotenv()
st.set_page_config(page_title="Fake Job Detector", page_icon="🛡️", layout="wide")


@st.cache_resource
def get_detector() -> HybridDetector:
    return HybridDetector()


def render_result(result: dict) -> None:
    col1, col2 = st.columns([1, 1])
    with col1:
        pred = result["prediction"].upper()
        st.metric("Prediction", pred)
        st.metric("Fake Confidence", f"{result['confidence_fake']:.2%}")

    with col2:
        st.subheader("Suspicious Signals")
        if result["rule_findings"]:
            for finding in result["rule_findings"]:
                st.write(f"- {finding}")
        else:
            st.write("- No explicit rule-based signal triggered.")

    st.subheader("Retrieved Similar Examples")
    for i, item in enumerate(result["similar_examples"], start=1):
        lbl = "fake" if item["label"] == 1 else "real"
        st.write(f"**#{i}** | label: `{lbl}` | similarity: `{item['similarity']:.3f}`")
        st.caption(item["text"])

    st.subheader("Formal Explanation")
    st.write(result["explanation"])


def main() -> None:
    st.title("Fake Job Detection Assistant")
    st.write(
        "Paste a job listing to detect fraud risk with a hybrid pipeline "
        "(ML + rules + retrieval + local LLM explanation)."
    )

    if "history" not in st.session_state:
        st.session_state.history = []

    with st.form("job_form"):
        job_text = st.text_area("Job Listing Text", height=260, placeholder="Paste full job listing here...")
        company_profile = st.text_input("Company Profile (optional)")
        salary_range = st.text_input("Salary Range (optional)")
        contact_info = st.text_input("Contact Info (optional)")
        submitted = st.form_submit_button("Analyze Job")

    if submitted:
        if not job_text.strip():
            st.error("Please provide the job listing text.")
            return

        detector = get_detector()
        with st.spinner("Running classification, retrieval, and explanation..."):
            result = detector.predict(
                job_text=job_text,
                company_profile=company_profile,
                salary_range=salary_range,
                contact_info=contact_info,
            )
        st.session_state.history.append({"input": job_text[:160], "result": result})
        render_result(result)

    st.divider()
    st.subheader("Session History")
    if not st.session_state.history:
        st.caption("No analyses yet.")
    else:
        for idx, entry in enumerate(reversed(st.session_state.history), start=1):
            st.write(f"**Run {idx}:** {entry['input']}...")
            st.caption(
                f"Prediction: {entry['result']['prediction']} | "
                f"Confidence(fake): {entry['result']['confidence_fake']:.2%}"
            )


if __name__ == "__main__":
    main()
