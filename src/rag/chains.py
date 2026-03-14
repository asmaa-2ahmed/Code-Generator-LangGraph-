from __future__ import annotations

from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from src.config import (
    SYSTEM_CODE,
    SYSTEM_EXPLAIN,
    SYSTEM_INTENT,
    make_llm,
)

# ====== LLM instances =========================================
llm_code    = make_llm("code")          
llm_explain = make_llm("explain")        
llm_meta    = make_llm("self_learning")  

# ====== Chains =================================================
_RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_CODE),
    MessagesPlaceholder(variable_name="history"),
    ("human", """
        Use the following similar examples as guidance:

        {context}

        Now solve this problem:
        {input}
            """),
])

rag_chain = _RAG_PROMPT | llm_code | StrOutputParser()

# -------------------------------------------------------------
_EXPLAIN_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_EXPLAIN),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
])

explain_chain = _EXPLAIN_PROMPT | llm_explain | StrOutputParser()

# -------------------------------------------------------------

_INTENT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_INTENT),
    ("human", "{input}"),
])

intent_chain = _INTENT_PROMPT | llm_meta | StrOutputParser()
