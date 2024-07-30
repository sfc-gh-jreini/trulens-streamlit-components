import streamlit as st
from trulens_eval import streamlit as trulens_st
from trulens_eval import Tru

from base import rag, filtered_rag, tru_rag, filtered_tru_rag, db_url

st.set_page_config(
    page_title="Use TruLens in Streamlit",
    page_icon="🦑",
)

st.title("TruLens ❤️ Streamlit")

st.write("Chat with the Streamlit docs, and view tracing & evaluation metrics powered by TruLens 🦑.")

tru = Tru(database_url=db_url)

with_filters = st.toggle("Use [Context Filter Guardrails](%s)" % "https://www.trulens.org/trulens_eval/guardrails/", value=False)

def generate_response(input_text):
    if with_filters:
        with filtered_tru_rag as recording:
            response = filtered_rag.query(input_text)
    else:
        with tru_rag as recording:
            response = rag.query(input_text)

    record = recording.get()
    
    return record, response

with st.form("my_form"):
    text = st.text_area(
        "Enter text:", "How do I launch a streamlit app?"
    )
    submitted = st.form_submit_button("Submit")
    if submitted:
        record, response = generate_response(text)
        st.info(response)

if submitted:
    with st.expander("See the trace of this record 👀"):
        trulens_st.trulens_trace(record=record)
    trulens_st.trulens_feedback(record=record)

with st.expander("Open to see aggregate evaluation metrics"):

    st.title("Aggregate Evaluation Metrics")
    st.write("Powered by TruLens 🦑.")

    trulens_st.trulens_leaderboard()

