import streamlit as st
from dotenv import load_dotenv
from typing import Iterator

from src.inference.predict_and_explain import HybridDetector


load_dotenv()
st.set_page_config(page_title="Fake Job Detector", page_icon="🛡️", layout="wide")


@st.cache_resource
def get_detector() -> HybridDetector:
    return HybridDetector()


def render_setup_error(exc: Exception) -> None:
    st.error("Model artifacts are not ready yet, so analysis cannot run.")
    st.info("Generate artifacts once, then restart the app.")
    st.code(
        "\n".join(
            [
                "python -m src.data.preprocess",
                "python -m src.models.train_baseline",
                "python -m src.retrieval.build_faiss_index",
            ]
        )
    )
    st.caption(f"Details: {exc}")


def _tone_prefix(tone: str) -> str:
    mapping = {
        "Formal": "Thank you for sharing the listing.",
        "Friendly": "Thanks for sending this over.",
        "Professor": "Good submission. Let us evaluate it methodically.",
    }
    return mapping.get(tone, "Thanks for sharing this listing.")


def _stream_chunks(text: str) -> Iterator[str]:
    words = text.split(" ")
    for i, word in enumerate(words):
        suffix = " " if i < len(words) - 1 else ""
        yield word + suffix


def format_human_response(result: dict) -> str:
    prediction = result["prediction"].upper()
    fake_conf = result["confidence_fake"]
    real_conf = 1.0 - fake_conf
    verdict_text = (
        "This posting looks suspicious and likely fake."
        if result["prediction"] == "fake"
        else "This posting looks likely legitimate."
    )
    signals = result["rule_findings"] or ["No major rule-based suspicious signal triggered."]
    signals_sentence = " ".join([f"{idx + 1}) {item}" for idx, item in enumerate(signals[:4])])

    return "\n".join(
        [
            f"{_tone_prefix('Formal')} I analyzed your posting and my verdict is **{prediction}**.",
            "",
            (
                f"{verdict_text} The model gives **fake probability = {fake_conf:.2%}** and "
                f"**real probability = {real_conf:.2%}**, and the decision rule is: "
                "if fake probability is at least 50%, classify as fake."
            ),
            "",
            "From the calculation side, the key risk signals are: "
            f"{signals_sentence}",
            "",
            "In plain language, here is the final reasoning:",
            f"{result['explanation']}",
        ]
    )


def render_calculation_details(result: dict) -> None:
    with st.expander("See calculation and evidence details"):
        col1, col2 = st.columns([1, 1])
        with col1:
            st.metric("Prediction", result["prediction"].upper())
            st.metric("Fake Probability", f"{result['confidence_fake']:.2%}")
            st.caption("Rule: fake if probability >= 50%")

        with col2:
            st.write("**Rule-based findings**")
            if result["rule_findings"]:
                for finding in result["rule_findings"]:
                    st.write(f"- {finding}")
            else:
                st.write("- No explicit suspicious signal triggered.")

        st.write("**Similar cases retrieved from dataset**")
        if not result["similar_examples"]:
            st.caption("No similar examples were retrieved.")
        else:
            for idx, item in enumerate(result["similar_examples"], start=1):
                label = "fake" if item["label"] == 1 else "real"
                st.write(f"- Example {idx}: label={label}, similarity={item['similarity']:.3f}")
                st.caption(item["text"])


def main() -> None:
    st.title("Fake Job Detection Assistant")
    st.write(
        "Chat with the assistant by pasting a full job description and extra details. "
        "You will get a human-style analysis with verdict, confidence, and evidence."
    )

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    "Hello, how are you? Paste the job description with any additional details. "
                    "I will explain clearly whether it looks fake or real and why."
                ),
            }
        ]
    if "stream_next_assistant" not in st.session_state:
        st.session_state.stream_next_assistant = False

    with st.form("chat_input_form"):
        user_input = st.text_area(
            "Your message / job posting",
            height=220,
            placeholder=(
                "Paste the full job listing here with any extra details "
                "(company info, salary, contact, benefits, etc.)."
            ),
        )
        submitted = st.form_submit_button("Send")

    if submitted:
        if not user_input.strip():
            st.warning("Please paste a job description first.")
        else:
            st.session_state.messages.append({"role": "user", "content": user_input})
            try:
                detector = get_detector()
            except Exception as exc:
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": "I cannot run the analysis yet because required artifacts are missing.",
                        "error": str(exc),
                    }
                )
                st.session_state.stream_next_assistant = True
            else:
                with st.spinner("Analyzing posting with ML, rules, retrieval, and LLM explanation..."):
                    result = detector.predict(job_text=user_input)
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": format_human_response(result),
                        "result": result,
                    }
                )
                st.session_state.stream_next_assistant = True

    st.subheader("Chat")
    last_idx = len(st.session_state.messages) - 1
    for idx, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            should_stream = (
                st.session_state.stream_next_assistant
                and idx == last_idx
                and message["role"] == "assistant"
            )
            if should_stream:
                st.write_stream(_stream_chunks(message["content"]))
                st.session_state.stream_next_assistant = False
            else:
                st.write(message["content"])
            if "result" in message:
                render_calculation_details(message["result"])
            if "error" in message:
                render_setup_error(Exception(message["error"]))


if __name__ == "__main__":
    main()
