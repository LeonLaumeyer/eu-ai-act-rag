"""Server entry point: ``uvicorn aiact_rag.main:create --factory``."""

from fastapi import FastAPI

from aiact_rag.api import create_app
from aiact_rag.config import load_config
from aiact_rag.pipeline import Resources, build_answer_fn, build_llm, build_retriever


def create() -> FastAPI:
    config = load_config()
    retriever = build_retriever(config.retrieval, Resources())
    return create_app(build_answer_fn(retriever, build_llm(config.llm)))
