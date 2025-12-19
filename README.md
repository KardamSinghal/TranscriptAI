# AI Transcript App

An AI-powered voice transcription application designed as a strong portfolio project for AI/ML and AI Engineering roles.

This application provides browser-based voice recording, local speech-to-text transcription using Whisper, and transcript refinement using a Large Language Model (LLM). The system is built with a FastAPI backend and a modern frontend interface.

---

## Project Overview

The AI Transcript App allows users to record audio directly from their browser, transcribe spoken English into text, and clean the transcript automatically by removing filler words, correcting grammar, and improving readability.

The project is fully modular and runs locally by default, making it beginner-friendly while still being extensible for advanced AI engineering use cases.

---

## Repository Structure & Branches

This repository is organized using checkpoint-based branches to demonstrate progressive AI engineering concepts.

| Branch | Description | Builds On |
|------|------------|-----------|
| `main` | Complete transcription app with Whisper + LLM cleaning (fully local setup) | — |
| `checkpoint-1-fundamentals` | Exercise generation system for Python and TypeScript fundamentals | — |
| `checkpoint-agentic-openrouter` | Agentic workflow with autonomous tool selection | `main` |
| `checkpoint-pydanticai-openrouter` | Structured agent development using PydanticAI | Previous checkpoint |
| `checkpoint-rest-mcp-openrouter` | MCP integration with REST APIs and GitHub-style issue workflows | Previous checkpoint |

Switch branches using:
```bash
git checkout <branch-name>
