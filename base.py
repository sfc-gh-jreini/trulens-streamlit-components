from typing import List

from snowflake.snowpark.session import Session
from snowflake.core import Root
from snowflake.cortex import Complete

import streamlit as st

from trulens_eval import Tru
from trulens_eval.guardrails.base import context_filter
from trulens_eval.tru_custom_app import TruCustomApp
from trulens_eval.tru_custom_app import instrument
from trulens_eval.feedback.provider.cortex import Cortex
from trulens_eval.feedback import Feedback
from trulens_eval import Select
import numpy as np

connection_details = {
  "account":  st.secrets["SNOWFLAKE_ACCOUNT"],
  "user": st.secrets["SNOWFLAKE_USER"],
  "password": st.secrets["SNOWFLAKE_USER_PASSWORD"],
  "role": st.secrets["SNOWFLAKE_ROLE"],
  "database": st.secrets["SNOWFLAKE_DATABASE"],
  "schema": st.secrets["SNOWFLAKE_SCHEMA"],
  "warehouse": st.secrets["SNOWFLAKE_WAREHOUSE"]
}

db_url = "snowflake://{user}:{password}@{account}/{dbname}/{schema}?warehouse={warehouse}&role={role}".format(
  user=st.secrets["SNOWFLAKE_USER"],
  account=st.secrets["SNOWFLAKE_ACCOUNT"],
  password=st.secrets["SNOWFLAKE_USER_PASSWORD"],
  dbname=st.secrets["SNOWFLAKE_DATABASE"],
  schema=st.secrets["SNOWFLAKE_SCHEMA"],
  warehouse=st.secrets["SNOWFLAKE_WAREHOUSE"],
  role=st.secrets["SNOWFLAKE_ROLE"],
)

session = Session.builder.configs(connection_details).create()

class CortexSearchRetriever:

    def __init__(self, session: Session, limit_to_retrieve: int = 4):
        self._session = session
        self._limit_to_retrieve = limit_to_retrieve

    def retrieve(self, query: str) -> List[str]:
        root = Root(self._session)
        cortex_search_service = (
        root
        .databases[st.secrets["SNOWFLAKE_DATABASE"]]
        .schemas[st.secrets["SNOWFLAKE_SCHEMA"]]
        .cortex_search_services[st.secrets["SNOWFLAKE_CORTEX_SEARCH_SERVICE"]]
    )
        resp = cortex_search_service.search(
                query=query,
                columns=["doc_text"],
                limit=self._limit_to_retrieve,
            )

        if resp.results:
            return [curr["doc_text"] for curr in resp.results]
        else:
            return []
        
provider = Cortex("mistral-large")

f_groundedness = (
    Feedback(
    provider.groundedness_measure_with_cot_reasons, name="Groundedness")
    .on(Select.RecordCalls.retrieve_context.rets[:].collect())
    .on_output()
)

f_context_relevance = (
    Feedback(
    provider.context_relevance,
    name="Context Relevance")
    .on_input()
    .on(Select.RecordCalls.retrieve_context.rets[:])
    .aggregate(np.mean)
)

f_answer_relevance = (
    Feedback(
    provider.relevance,
    name="Answer Relevance")
    .on_input()
    .on_output()
    .aggregate(np.mean)
)

feedbacks = [f_context_relevance,
            f_answer_relevance,
            f_groundedness,
        ]

class RAG_from_scratch:

  def __init__(self):
    self.retriever = CortexSearchRetriever(session=session, limit_to_retrieve=4)

  @instrument
  def retrieve_context(self, query: str) -> list:
    """
    Retrieve relevant text from vector store.
    """
    return self.retriever.retrieve(query)

  @instrument
  def generate_completion(self, query: str, context_str: list) -> str:
    """
    Generate answer from context.
    """
    prompt = f"""
    'You are an expert assistance extracting information from context provided.
    Answer the question based on the context. Be concise and do not hallucinate.
    If you don´t have the information just say so.
    Context: {context_str}
    Question:
    {query}
    Answer: '
    """
    return Complete("mistral-large", query)

  @instrument
  def query(self, query: str) -> str:
    context_str = self.retrieve_context(query)
    return self.generate_completion(query, context_str)

class filtered_RAG_from_scratch:

    def __init__(self):
        self.retriever = CortexSearchRetriever(session=session, limit_to_retrieve=4)

    @instrument
    @context_filter(f_context_relevance, 0.75, keyword_for_prompt="query")
    def retrieve_context(self, query: str) -> list:
        """
        Retrieve relevant text from vector store.
        """
        results = self.retriever.retrieve(query)
        return results

    @instrument
    def generate_completion(self, query: str, context_str: list) -> str:
        """
        Generate answer from context.
        """
        completion = Complete("mistral-large",query)
        return completion

    @instrument
    def query(self, query: str) -> str:
        context_str = self.retrieve_context(query=query)
        completion = self.generate_completion(query=query, context_str=context_str)
        return completion
    
rag = RAG_from_scratch()

tru_rag = TruCustomApp(rag,
    app_id = 'RAG v1',
    feedbacks = [f_groundedness, f_answer_relevance, f_context_relevance])

filtered_rag = filtered_RAG_from_scratch()

filtered_tru_rag = TruCustomApp(filtered_rag,
    app_id = 'Filtered RAG App',
    feedbacks = [f_groundedness, f_answer_relevance, f_context_relevance])