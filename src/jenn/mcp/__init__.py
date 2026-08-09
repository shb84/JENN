"""JENN MCP server package.
===========================

Local stdio MCP server exposing JENN surrogate-modeling tools. Requires the
optional ``mcp`` dependency (``pip install "jenn[mcp]"``).

This ``__init__`` deliberately imports nothing, so ``import jenn.mcp`` stays
cheap and never forces the optional dependency. The server itself lives in
:mod:`jenn.mcp.server`.
"""

# Copyright (C) 2018 Steven H. Berguin
# This work is licensed under the MIT License.
