"""Compatibility shim so RAGAS 0.4.3 imports with langchain-community 0.4.x.

RAGAS imports `langchain_community.chat_models.vertexai.ChatVertexAI` at import time, but that module no longer
exists in langchain-community 0.4.x (and 0.3.x needs langchain-core < 1, which this project does not use).
RAGAS only uses the class for Vertex-specific isinstance checks, so a placeholder is enough. Import this
module BEFORE importing ragas. It affects only the evaluation scripts, not the application.
"""
import sys
import types


def apply() -> None:
    for name, attr in (("langchain_community.chat_models.vertexai", "ChatVertexAI"),
                       ("langchain_community.llms.vertexai", "VertexAI")):
        try:
            __import__(name)
        except ImportError:
            module = types.ModuleType(name)
            setattr(module, attr, type(attr, (), {}))
            sys.modules[name] = module


apply()
