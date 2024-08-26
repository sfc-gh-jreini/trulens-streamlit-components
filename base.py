from typing import List

from snowflake.snowpark.session import Session
from snowflake.core import Root
from snowflake.cortex import Complete

import streamlit as st

from trulens.core import TruSession
from trulens.core.guardrails.base import context_filter
from trulens.core import TruCustomApp
from trulens.core.app.custom import instrument
from trulens.providers.cortex import Cortex
from trulens.core import Feedback
from trulens.core import Select
import numpy as np

from sqlalchemy import create_engine
from snowflake.sqlalchemy import URL

from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization

p_key= serialization.load_pem_private_key(
    st.secrets["SNOWFLAKE_PRIVATE_KEY"].encode(),
    password=None,
    backend=default_backend()
    )

pkb = p_key.private_bytes(
    encoding=serialization.Encoding.DER,
    format=serialization.PrivateFormat.PKCS8,
    encryption_algorithm=serialization.NoEncryption())

connection_details = {
  "account":  st.secrets["SNOWFLAKE_ACCOUNT"],
  "user": st.secrets["SNOWFLAKE_USER"],
  "private_key": pkb,
  "role": st.secrets["SNOWFLAKE_ROLE"],
  "database": st.secrets["SNOWFLAKE_DATABASE"],
  "schema": st.secrets["SNOWFLAKE_SCHEMA"],
  "warehouse": st.secrets["SNOWFLAKE_WAREHOUSE"]
}

engine = create_engine(URL(
    account=st.secrets["SNOWFLAKE_ACCOUNT"],
    warehouse=st.secrets["SNOWFLAKE_WAREHOUSE"],
    database=st.secrets["SNOWFLAKE_DATABASE"],
    schema=st.secrets["SNOWFLAKE_SCHEMA"],
    user=st.secrets["SNOWFLAKE_USER"],),
    connect_args={
            'private_key': pkb,
            },
    )

tru = TruSession(database_engine = engine)
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
        
provider = Cortex("llama3.1-8b")

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
    feedbacks = feedbacks)

filtered_rag = filtered_RAG_from_scratch()

filtered_tru_rag = TruCustomApp(filtered_rag,
    app_id = 'Filtered RAG App',
    feedbacks = feedbacks)
