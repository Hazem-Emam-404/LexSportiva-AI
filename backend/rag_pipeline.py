import os
import sys
import asyncio
from typing import List, Dict, Optional, Any, Tuple
from dotenv import load_dotenv

load_dotenv()

from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain_community.retrievers import BM25Retriever
from sentence_transformers import CrossEncoder

from utils import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_CROSS_ENCODER_MODEL,
    DEFAULT_BM25_K,
    DEFAULT_TOP_K_CHILDREN,
    DEFAULT_TOP_N_RERANK,
    DEFAULT_RRF_K,
    DEFAULT_FUSED_CANDIDATES_K,
    DEFAULT_LLM_TEMPERATURE,
    SPORT_FILE_MAP,
    map_sport_to_filename,
    format_chat_history,
    chunking_embedding,
    DECOMPOSER_SYSTEM_PROMPT,
    SPORTS_REFERENCE_SYSTEM_PROMPT,
)

# ---------------------------------------------------------------------------
# Pydantic Schemas for Structured Output
# ---------------------------------------------------------------------------
class ClaimSource(BaseModel):
    file_name: str = Field(
        description="Rulebook name the claim came from, e.g. 'Football', 'Basketball', 'Handball', 'Tennis', 'Boxing'"
    )
    page_number: str = Field(
        description="Page number in the rulebook where this claim appears"
    )


class Claim(BaseModel):
    claim: str = Field(
        description="A specific factual statement taken from the context that supports part of the answer"
    )
    source: ClaimSource = Field(
        description="Exact source (rulebook + page) this claim was extracted from"
    )


class RefereeAnswer(BaseModel):
    answer: str = Field(
        description="Final answer to the user, written in the same language the user asked in"
    )
    claims: List[Claim] = Field(
        default_factory=list,
        description="Every factual claim backing the answer, each tagged with its source. MUST be empty if the answer is a greeting/chitchat, or if no rule was found."
    )


class SubQuestion(BaseModel):
    question: str = Field(
        description="Atomic, self-contained sub-question in English."
    )
    sport: Optional[str] = Field(
        default=None,
        description="The specific sport this question targets ('Football', 'Basketball', 'Handball', 'Tennis', 'Boxing'). If generic across multiple sports, cannot be specified, or not about sports, set to None."
    )


class QueryAnalysis(BaseModel):
    detected_language: str = Field(
        description="The language of the original user query, e.g. 'Arabic', 'English', 'French'"
    )
    translated_query: str = Field(
        description="The full original query translated into English, preserving its complete meaning and intent"
    )
    sub_questions: List[SubQuestion] = Field(
        description="Atomic, self-contained sub-questions in English with their specified sport (or None if generic/unspecified)."
    )


# (System prompts DECOMPOSER_SYSTEM_PROMPT and SPORTS_REFERENCE_SYSTEM_PROMPT are imported from utils)


# ---------------------------------------------------------------------------
# Sports RAG Pipeline Class
# ---------------------------------------------------------------------------
class SportsRAGPipeline:
    """
    Production-ready Multi-Sport RAG Engine.
    Handles query decomposition, multi-turn coreference resolution,
    sport-filtered hybrid retrieval (dense + BM25 + CrossEncoder rerank),
    parent-child doc resolution, and grounded answer generation with claim citations.
    """

    def __init__(
        self,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        cross_encoder_model: str = DEFAULT_CROSS_ENCODER_MODEL,
        decomposer_model: str = "openai/gpt-oss-20b",
        generator_model: str = "qwen/qwen3.8-27b",
        groq_api_key: Optional[str] = None,
        bm25_k: int = DEFAULT_BM25_K
    ):
        self.api_key = groq_api_key or os.environ.get("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY must be set in environment or passed to SportsRAGPipeline.")

        # 1. Load Vectorstore & Docstore
        print("Initializing SportsRAGPipeline components...")
        self.retriever, self.vectorstore, self.store = chunking_embedding(
            embedding_model=embedding_model,
            loaded_files=None,
            is_for_embedding=False
        )

        # 2. Initialize BM25 Retriever
        chroma_data = self.vectorstore.get()
        child_docs = [
            Document(page_content=content, metadata=metadata)
            for content, metadata in zip(chroma_data["documents"], chroma_data["metadatas"])
        ]
        self.bm25_retriever = BM25Retriever.from_documents(child_docs)
        self.bm25_retriever.k = bm25_k

        # 3. Cross-Encoder Reranker
        self.reranker = CrossEncoder(cross_encoder_model)

        # 4. LLMs
        self.decomposer_llm = ChatGroq(
            model=decomposer_model,
            temperature=DEFAULT_LLM_TEMPERATURE,
            api_key=self.api_key
        )
        self.structured_decomposer = self.decomposer_llm.with_structured_output(QueryAnalysis)
        self.decomposer_prompt = ChatPromptTemplate.from_messages([
            ("system", DECOMPOSER_SYSTEM_PROMPT),
            ("human", "{query}")
        ])
        self.decomposer_chain = self.decomposer_prompt | self.structured_decomposer

        self.generator_llm = ChatGroq(
            model=generator_model,
            temperature=DEFAULT_LLM_TEMPERATURE,
            api_key=self.api_key
        )
        self.structured_generator = self.generator_llm.with_structured_output(RefereeAnswer)
        self.sports_prompt = ChatPromptTemplate.from_messages([
            ("system", SPORTS_REFERENCE_SYSTEM_PROMPT),
            ("human", "User Query / Situation: {query}")
        ])
        self.sports_rule_chain = self.sports_prompt | self.structured_generator
        print("SportsRAGPipeline ready.")

    def advanced_parent_child_retrieval(
        self,
        query: str,
        top_k_children: int = DEFAULT_TOP_K_CHILDREN,
        top_n_rerank: int = DEFAULT_TOP_N_RERANK,
        sport: Optional[str] = None
    ) -> List[Document]:
        """
        Thread-safe hybrid retrieval with sport-level metadata filtering,
        RRF fusion, Cross-Encoder reranking, and parent document resolution.
        """
        target_pdf = map_sport_to_filename(sport)
        target_file_name = target_pdf.replace(".pdf", "") if target_pdf else None

        # 1. Dense retrieval with metadata filtering
        if target_file_name:
            dense_children = self.vectorstore.similarity_search(
                query,
                k=top_k_children,
                filter={"file_name": target_file_name}
            )
        else:
            dense_children = self.vectorstore.similarity_search(query, k=top_k_children)

        # 2. Sparse (BM25) retrieval (Thread-safe: slice list without modifying self.bm25_retriever.k)
        all_sparse = self.bm25_retriever.invoke(query)
        if target_file_name:
            sparse_children = [
                d for d in all_sparse
                if d.metadata.get("file_name") == target_file_name
            ][:top_k_children]
        else:
            sparse_children = all_sparse[:top_k_children]

        # 3. Reciprocal Rank Fusion (RRF)
        fused_scores: Dict[str, float] = {}
        unique_children_map: Dict[str, Document] = {}

        for rank, doc in enumerate(dense_children):
            content = doc.page_content
            fused_scores[content] = fused_scores.get(content, 0.0) + (1.0 / (rank + DEFAULT_RRF_K))
            unique_children_map[content] = doc

        for rank, doc in enumerate(sparse_children):
            content = doc.page_content
            fused_scores[content] = fused_scores.get(content, 0.0) + (1.0 / (rank + DEFAULT_RRF_K))
            unique_children_map[content] = doc

        fused_candidates = sorted(
            unique_children_map.values(),
            key=lambda d: fused_scores[d.page_content],
            reverse=True
        )[:DEFAULT_FUSED_CANDIDATES_K]

        if not fused_candidates:
            return []

        # 4. Cross-Encoder Reranking
        cross_inputs = [[query, doc.page_content] for doc in fused_candidates]
        rerank_scores = self.reranker.predict(cross_inputs)
        for doc, score in zip(fused_candidates, rerank_scores):
            doc.metadata["rerank_score"] = float(score)

        top_n_children = sorted(
            fused_candidates,
            key=lambda d: d.metadata.get("rerank_score", 0.0),
            reverse=True
        )[:top_n_rerank]

        # 5. Parent Document Resolution from DocStore
        unique_parent_ids = []
        seen_ids = set()
        for child in top_n_children:
            pid = child.metadata.get("doc_id")
            if pid and pid not in seen_ids:
                seen_ids.add(pid)
                unique_parent_ids.append(pid)

        resolved_parents = self.store.mget(unique_parent_ids)
        final_context = []
        for pid, doc in zip(unique_parent_ids, resolved_parents):
            if doc is not None:
                doc.metadata["doc_id"] = pid
                final_context.append(doc)

        return final_context

    async def retrieve_with_decomposition(
        self,
        user_query: str,
        chat_history: Optional[List[Dict[str, str]]] = None
    ) -> Tuple[List[Document], QueryAnalysis]:
        """
        Decomposes query using chat history into self-contained sub-questions,
        and retrieves parent docs concurrently in background threads.
        """
        formatted_history = format_chat_history(chat_history)
        analysis: QueryAnalysis = await self.decomposer_chain.ainvoke({
            "query": user_query,
            "chat_history": formatted_history
        })

        # Parallel retrieval across sub-questions
        tasks = [
            asyncio.to_thread(
                self.advanced_parent_child_retrieval,
                query=sq.question,
                sport=sq.sport
            )
            for sq in analysis.sub_questions
        ]
        results = await asyncio.gather(*tasks)

        all_docs = []
        for docs in results:
            all_docs.extend(docs)

        # Deduplicate parent docs across sub-questions
        seen = set()
        unique_docs = []
        for doc in all_docs:
            doc_id = doc.metadata.get("doc_id")
            if doc_id not in seen:
                seen.add(doc_id)
                unique_docs.append(doc)

        return unique_docs, analysis

    async def query(
        self,
        query: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
        return_sources: bool = False
    ) -> Dict[str, Any] | RefereeAnswer:
        """
        End-to-end multi-turn RAG reasoning.
        """
        retrieved_docs, analysis = await self.retrieve_with_decomposition(
            query,
            chat_history=chat_history
        )

        if not retrieved_docs:
            msg = (
                "لم يتم العثور على أي نصوص أو قوانين ذات صلة في كتب القواعد الرسمية لهذه المسألة."
                if getattr(analysis, "detected_language", "").lower() == "arabic"
                else "No relevant rulebook sections found for this query in the official rulebooks."
            )
            if return_sources:
                return {
                    "answer": msg,
                    "claims": [],
                    "sources": [],
                    "analysis": analysis
                }
            return RefereeAnswer(answer=msg, claims=[])

        # Format context with source metadata
        formatted_context = "\n\n---\n\n".join([
            f"[{doc.metadata.get('file_name', 'Rulebook')} | Page {doc.metadata.get('page_number', 'N/A')}]:\n{doc.page_content}"
            for doc in retrieved_docs
        ])
        formatted_history = format_chat_history(chat_history)

        # Grounded reasoning generation
        response: RefereeAnswer = await self.sports_rule_chain.ainvoke({
            "query": query,
            "context": formatted_context,
            "chat_history": formatted_history
        })

        if return_sources:
            return {
                "answer": response.answer,
                "claims": response.claims,
                "sources": retrieved_docs,
                "analysis": analysis
            }

        return response

    def query_sync(
        self,
        query: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
        return_sources: bool = False
    ) -> Dict[str, Any] | RefereeAnswer:
        """Synchronous wrapper for pipeline query."""
        return asyncio.run(
            self.query(query, chat_history=chat_history, return_sources=return_sources)
        )


# ---------------------------------------------------------------------------
# Global Singleton & Module-Level Convenience Functions
# ---------------------------------------------------------------------------
_GLOBAL_PIPELINE: Optional[SportsRAGPipeline] = None


def get_pipeline() -> SportsRAGPipeline:
    """Returns or lazily creates the global SportsRAGPipeline singleton."""
    global _GLOBAL_PIPELINE
    if _GLOBAL_PIPELINE is None:
        _GLOBAL_PIPELINE = SportsRAGPipeline()
    return _GLOBAL_PIPELINE


async def ask_sports_rules(
    query: str,
    chat_history: Optional[List[Dict[str, str]]] = None,
    return_sources: bool = False
):
    """
    Main async function for answering sports rules questions.
    Uses 'await' and supports multi-turn chat history.
    """
    return await get_pipeline().query(
        query, chat_history=chat_history, return_sources=return_sources
    )


def ask_sports_rules_sync(
    query: str,
    chat_history: Optional[List[Dict[str, str]]] = None,
    return_sources: bool = False
):
    """
    Synchronous version of ask_sports_rules (no 'await' required).
    Ideal for regular Python scripts, standard loops, or simple testing.
    """
    return get_pipeline().query_sync(
        query, chat_history=chat_history, return_sources=return_sources
    )


# ---------------------------------------------------------------------------
# Stateful Chat Session Wrapper
# ---------------------------------------------------------------------------
class SportsChatSession:
    """
    Manages conversational memory across multiple turns.
    """

    def __init__(self, pipeline: Optional[SportsRAGPipeline] = None):
        self.pipeline = pipeline or get_pipeline()
        self.history: List[Dict[str, str]] = []

    async def chat(self, user_query: str, return_sources: bool = True) -> Dict[str, Any]:
        res = await self.pipeline.query(
            user_query,
            chat_history=self.history,
            return_sources=return_sources
        )

        answer_text = res["answer"] if isinstance(res, dict) else (res.answer if hasattr(res, "answer") else str(res))

        self.history.append({"role": "user", "content": user_query})
        self.history.append({"role": "assistant", "content": answer_text})
        return res

    def chat_sync(self, user_query: str, return_sources: bool = True) -> Dict[str, Any]:
        return asyncio.run(self.chat(user_query, return_sources=return_sources))

    def reset(self):
        self.history.clear()
        print("Chat history reset.")


# ---------------------------------------------------------------------------
# Standalone CLI Self-Test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")

    async def _test():
        pipeline = get_pipeline()
        print("\n--- Running Test Query ---")
        q = "ما هو حكم التسلل في كرة القدم؟"
        res = await pipeline.query(q, return_sources=True)
        print("\nAnswer:\n", res["answer"])
        print("\nClaims:")
        for c in res["claims"]:
            print(f"- {c.claim} [{c.source.file_name} p.{c.source.page_number}]")

    asyncio.run(_test())
