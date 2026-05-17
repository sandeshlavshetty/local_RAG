from __future__ import annotations

import json
import os
import re
from typing import Any, Callable, Dict, List, Literal, Tuple, TypedDict

import ollama
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph


ChunkResult = Tuple[str, Dict[str, Any]]
RetrieverFn = Callable[[str, int, str], List[ChunkResult]]


class RAGAgentState(TypedDict, total=False):
	original_query: str
	query: str
	thread_id: str
	thread_notes: str
	retrieval_method: str
	top_k: int
	retrieval_hint: str
	should_retrieve: bool
	rewrite_count: int
	max_rewrites: int
	retrieved_chunks: List[ChunkResult]
	citations: List[Dict[str, Any]]
	context_text: str
	is_relevant: bool
	answer: str
	query_used: str
	error: str
	turn_summary: str


def _safe_json_parse(raw: str) -> Dict[str, Any]:
	try:
		return json.loads(raw)
	except Exception:
		pass

	# Extract JSON object if model wrapped it with extra text.
	match = re.search(r"\{[\s\S]*\}", raw)
	if not match:
		return {}

	try:
		return json.loads(match.group(0))
	except Exception:
		return {}


def _chat(model: str, system: str, user: str) -> str:
	response = ollama.chat(
		model=model,
		messages=[
			{"role": "system", "content": system},
			{"role": "user", "content": user},
		],
	)
	return response["message"]["content"]


class LangGraphRetrieverAgent:
	def __init__(self, retriever_fn: RetrieverFn):
		self._retriever_fn = retriever_fn
		self._model = os.getenv("OLLAMA_VL_MODEL", "qwen3:8b")
		print(f"[DEBUG AGENT] Initializing LangGraphRetrieverAgent with model={self._model}")
		self._checkpointer = self._build_checkpointer()
		self._graph = self._build_graph()
		print("[DEBUG AGENT] LangGraphRetrieverAgent initialization complete")

	def _build_checkpointer(self):
		mode = os.getenv("LANGGRAPH_CHECKPOINTER", "memory").strip().lower()
		print(f"[DEBUG AGENT] Requested checkpointer mode: {mode}")
		if mode == "postgres":
			db_uri = os.getenv("LANGGRAPH_POSTGRES_URI", "").strip()
			if db_uri:
				try:
					from langgraph.checkpoint.postgres import PostgresSaver

					checkpointer = PostgresSaver.from_conn_string(db_uri)
					if hasattr(checkpointer, "setup"):
						checkpointer.setup()
					print("[DEBUG AGENT] Using Postgres checkpointer")
					return checkpointer
				except Exception as exc:
					print(f"[WARNING RETRIEVER] Postgres checkpointer unavailable: {exc}")

		if mode == "redis":
			db_uri = os.getenv("LANGGRAPH_REDIS_URI", "").strip()
			if db_uri:
				try:
					from langgraph.checkpoint.redis import RedisSaver

					checkpointer = RedisSaver.from_conn_string(db_uri)
					if hasattr(checkpointer, "setup"):
						checkpointer.setup()
					print("[DEBUG AGENT] Using Redis checkpointer")
					return checkpointer
				except Exception as exc:
					print(f"[WARNING RETRIEVER] Redis checkpointer unavailable: {exc}")

		print("[DEBUG AGENT] Falling back to InMemorySaver checkpointer")
		return InMemorySaver()

	def _build_graph(self):
		print("[DEBUG AGENT] Building LangGraph state machine")
		graph = StateGraph(RAGAgentState)

		graph.add_node("decide_retrieve_or_respond", self._decide_retrieve_or_respond)
		graph.add_node("retrieve", self._retrieve)
		graph.add_node("grade_documents", self._grade_documents)
		graph.add_node("rewrite_question", self._rewrite_question)
		graph.add_node("generate_answer", self._generate_answer)
		graph.add_node("update_thread_memory", self._update_thread_memory)

		graph.add_edge(START, "decide_retrieve_or_respond")
		graph.add_conditional_edges(
			"decide_retrieve_or_respond",
			self._route_after_decision,
			{
				"retrieve": "retrieve",
				"end": "update_thread_memory",
			},
		)
		graph.add_edge("retrieve", "grade_documents")
		graph.add_conditional_edges(
			"grade_documents",
			self._route_after_grading,
			{
				"answer": "generate_answer",
				"rewrite": "rewrite_question",
			},
		)
		graph.add_edge("rewrite_question", "decide_retrieve_or_respond")
		graph.add_edge("generate_answer", "update_thread_memory")
		graph.add_edge("update_thread_memory", END)
		print("[DEBUG AGENT] Compiling LangGraph with configured checkpointer")
		return graph.compile(checkpointer=self._checkpointer)

	def _decide_retrieve_or_respond(self, state: RAGAgentState) -> RAGAgentState:
		query = state["query"]
		thread_notes = state.get("thread_notes", "")
		print(f"[DEBUG AGENT] decide_retrieve_or_respond: query_len={len(query)}, thread_notes_len={len(thread_notes)}")
		system_prompt = (
			"You are a retrieval planner for a local RAG system. "
			"Decide whether retrieval is needed before answering. "
			"Return only JSON with keys: action, retrieval_hint, direct_answer."
		)
		user_prompt = (
			"User question:\n"
			f"{query}\n\n"
			f"Conversation memory:\n{thread_notes if thread_notes else '(none)'}\n\n"
			"Rules:\n"
			"- action must be one of: retrieve or direct\n"
			"- retrieval_hint should be a concise retrieval query\n"
			"- if action is direct, provide direct_answer\n"
			"- do not include any text outside JSON"
		)

		raw = _chat(self._model, system_prompt, user_prompt)
		parsed = _safe_json_parse(raw)

		action = str(parsed.get("action", "retrieve")).strip().lower()
		retrieval_hint = str(parsed.get("retrieval_hint", "")).strip() or query
		direct_answer = str(parsed.get("direct_answer", "")).strip()
		print(f"[DEBUG AGENT] Planner action={action}, retrieval_hint_len={len(retrieval_hint)}, has_direct_answer={bool(direct_answer)}")

		if action == "direct" and direct_answer:
			return {
				"should_retrieve": False,
				"retrieval_hint": retrieval_hint,
				"answer": direct_answer,
				"query_used": retrieval_hint,
			}

		return {
			"should_retrieve": True,
			"retrieval_hint": retrieval_hint,
			"query_used": retrieval_hint,
		}

	def _route_after_decision(self, state: RAGAgentState) -> Literal["retrieve", "end"]:
		if state.get("should_retrieve"):
			print("[DEBUG AGENT] route_after_decision: retrieving context")
			return "retrieve"
		print("[DEBUG AGENT] route_after_decision: direct response, updating thread memory")
		return "end"

	def _retrieve(self, state: RAGAgentState) -> RAGAgentState:
		retrieval_hint = state.get("retrieval_hint") or state["query"]
		top_k = int(state.get("top_k", 3))
		method = state.get("retrieval_method", "hybrid")
		print(f"[DEBUG AGENT] retrieve: method={method}, top_k={top_k}, hint='{retrieval_hint[:120]}'")

		chunks = self._retriever_fn(retrieval_hint, top_k=top_k, method=method)
		print(f"[DEBUG AGENT] retrieve: fetched_chunks={len(chunks)}")

		context_blocks: List[str] = []
		citations: List[Dict[str, Any]] = []
		for chunk, meta in chunks:
			source = meta.get("source_file", "N/A")
			page = meta.get("page_num", "N/A")
			context_blocks.append(f"[Source: {source}, page: {page}]\n{chunk}")
			citations.append(
				{
					"text": chunk,
					"source_file": source,
					"page_num": page,
				}
			)

		return {
			"retrieved_chunks": chunks,
			"context_text": "\n\n".join(context_blocks),
			"citations": citations,
		}

	def _grade_documents(self, state: RAGAgentState) -> RAGAgentState:
		question = state["query"]
		context = state.get("context_text", "")
		print(f"[DEBUG AGENT] grade_documents: question_len={len(question)}, context_len={len(context)}")

		if not context.strip():
			print("[DEBUG AGENT] grade_documents: no context available, marking irrelevant")
			return {"is_relevant": False}

		system_prompt = (
			"You are a strict relevance grader for RAG retrieval. "
			"Return only JSON with key binary_score and value yes or no."
		)
		user_prompt = (
			"Question:\n"
			f"{question}\n\n"
			"Retrieved context:\n"
			f"{context}\n\n"
			"Grade whether the context is relevant to answer the question."
		)

		raw = _chat(self._model, system_prompt, user_prompt)
		parsed = _safe_json_parse(raw)
		score = str(parsed.get("binary_score", "no")).strip().lower()
		print(f"[DEBUG AGENT] grade_documents: binary_score={score}")

		return {"is_relevant": score == "yes"}

	def _route_after_grading(self, state: RAGAgentState) -> Literal["answer", "rewrite"]:
		if state.get("is_relevant"):
			print("[DEBUG AGENT] route_after_grading: relevant context, generating answer")
			return "answer"

		rewrite_count = int(state.get("rewrite_count", 0))
		max_rewrites = int(state.get("max_rewrites", 1))
		if rewrite_count < max_rewrites:
			print(f"[DEBUG AGENT] route_after_grading: rewriting query ({rewrite_count + 1}/{max_rewrites})")
			return "rewrite"

		print("[DEBUG AGENT] route_after_grading: rewrite limit reached, generating answer")
		return "answer"

	def _rewrite_question(self, state: RAGAgentState) -> RAGAgentState:
		original = state.get("original_query", state["query"])
		current = state["query"]
		retrieval_hint = state.get("retrieval_hint", "")

		system_prompt = "You rewrite questions to improve retrieval quality for a local RAG system."
		user_prompt = (
			"Rewrite the question for better retrieval while preserving user intent.\n"
			f"Original question: {original}\n"
			f"Current question: {current}\n"
			f"Current retrieval hint: {retrieval_hint}\n"
			"Return only the rewritten question text."
		)

		rewritten = _chat(self._model, system_prompt, user_prompt).strip()
		if not rewritten:
			rewritten = current
		print(f"[DEBUG AGENT] rewrite_question: old_len={len(current)}, new_len={len(rewritten)}")

		return {
			"query": rewritten,
			"retrieval_hint": rewritten,
			"rewrite_count": int(state.get("rewrite_count", 0)) + 1,
			"query_used": rewritten,
		}

	def _generate_answer(self, state: RAGAgentState) -> RAGAgentState:
		question = state.get("original_query", state["query"])
		context = state.get("context_text", "")
		thread_notes = state.get("thread_notes", "")
		print(f"[DEBUG AGENT] generate_answer: question_len={len(question)}, context_len={len(context)}, thread_notes_len={len(thread_notes)}")

		if not context.strip():
			print("[DEBUG AGENT] generate_answer: no context, returning fallback response")
			return {
				"answer": (
					"I could not find relevant context in your indexed data for this question. "
					"Try rephrasing the question or upload more related documents."
				)
			}

		system_prompt = (
			"You are a helpful assistant. Answer only from the provided context. "
			"If context is insufficient, say you do not know."
		)
		user_prompt = (
			f"Question: {question}\n\n"
			f"Conversation memory:\n{thread_notes if thread_notes else '(none)'}\n\n"
			f"Context:\n{context}\n\n"
			"Provide a concise answer."
		)

		answer = _chat(self._model, system_prompt, user_prompt).strip()
		print(f"[DEBUG AGENT] generate_answer: answer_len={len(answer)}")
		return {"answer": answer}

	def _update_thread_memory(self, state: RAGAgentState) -> RAGAgentState:
		question = state.get("original_query", state.get("query", ""))
		answer = state.get("answer", "")
		previous_notes = state.get("thread_notes", "")

		turn_summary = f"Q: {question}\nA: {answer}".strip()
		if previous_notes.strip():
			thread_notes = f"{previous_notes}\n\n{turn_summary}"
		else:
			thread_notes = turn_summary
		print(f"[DEBUG AGENT] update_thread_memory: turn_summary_len={len(turn_summary)}, total_notes_len={len(thread_notes)}")

		return {
			"thread_notes": thread_notes,
			"turn_summary": turn_summary,
		}

	def invoke(self, query: str, top_k: int = 3, retrieval_method: str = "hybrid") -> Dict[str, Any]:
		print(f"[DEBUG AGENT] invoke called: query_len={len(query)}, top_k={top_k}, method={retrieval_method}")
		initial_state: RAGAgentState = {
			"original_query": query,
			"query": query,
			"retrieval_method": retrieval_method,
			"top_k": top_k,
			"rewrite_count": 0,
			"max_rewrites": 1,
			"citations": [],
			"context_text": "",
		}
		final_state = self._graph.invoke(initial_state)
		print(f"[DEBUG AGENT] invoke completed: answer_len={len(final_state.get('answer', ''))}, citations={len(final_state.get('citations', []))}")
		return {
			"answer": final_state.get("answer", "No answer generated."),
			"citations": final_state.get("citations", []),
			"query_used": final_state.get("query_used", final_state.get("retrieval_hint", "")),
		}

	def _debug_thread_memory_snapshot(self, config: Dict[str, Any], stage: str) -> None:
		"""Best-effort snapshot of persisted thread memory for debugging."""
		try:
			snapshot = self._graph.get_state(config)
		except Exception as exc:
			print(f"[DEBUG AGENT] {stage}: unable to read checkpoint state: {exc}")
			return

		values = {}
		if isinstance(snapshot, dict):
			values = snapshot.get("values", {}) if isinstance(snapshot.get("values", {}), dict) else {}
		else:
			maybe_values = getattr(snapshot, "values", {})
			if isinstance(maybe_values, dict):
				values = maybe_values

		thread_notes = values.get("thread_notes", "") if isinstance(values, dict) else ""
		query = values.get("query", "") if isinstance(values, dict) else ""
		answer = values.get("answer", "") if isinstance(values, dict) else ""

		print(
			f"[DEBUG AGENT] {stage}: checkpoint snapshot "
			f"thread_notes_len={len(thread_notes)}, query_len={len(query)}, answer_len={len(answer)}"
		)
		if thread_notes:
			preview = thread_notes[:300].replace("\n", "\\n")
			print(f"[DEBUG AGENT] {stage}: thread_notes_preview={preview}")

	def invoke_thread(
		self,
		query: str,
		thread_id: str,
		top_k: int = 3,
		retrieval_method: str = "hybrid",
	) -> Dict[str, Any]:
		print(f"[DEBUG AGENT] invoke_thread called: thread_id={thread_id}, query_len={len(query)}, top_k={top_k}, method={retrieval_method}")
		initial_state: RAGAgentState = {
			"original_query": query,
			"query": query,
			"thread_id": str(thread_id),
			"retrieval_method": retrieval_method,
			"top_k": top_k,
			"rewrite_count": 0,
			"max_rewrites": 1,
			"citations": [],
			"context_text": "",
		}
		config = {"configurable": {"thread_id": str(thread_id)}}
		print(f"[DEBUG AGENT] invoke_thread using graph config: {config}")
		self._debug_thread_memory_snapshot(config, "before_invoke_thread")
		final_state = self._graph.invoke(initial_state, config)
		self._debug_thread_memory_snapshot(config, "after_invoke_thread")
		print(f"[DEBUG AGENT] invoke_thread completed: answer_len={len(final_state.get('answer', ''))}, citations={len(final_state.get('citations', []))}, thread_notes_len={len(final_state.get('thread_notes', ''))}")
		return {
			"answer": final_state.get("answer", "No answer generated."),
			"citations": final_state.get("citations", []),
			"query_used": final_state.get("query_used", final_state.get("retrieval_hint", "")),
			"thread_id": str(thread_id),
			"thread_notes": final_state.get("thread_notes", ""),
		}

