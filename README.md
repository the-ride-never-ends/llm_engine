
# LLM Engine Module

## Overview
The LLM Engine module provides a unified interface for interacting with various Language Model APIs and engines. 
It includes two main classes: LlmEngine for handling API interactions and GenerationStep for managing LLM generation tasks.
This is mainly derived from E.P. Armstrong's excellent Augmentoolkit, which you can find here: https://github.com/e-p-armstrong/augmentoolkit

## Key Features
- Support for multiple LLM backends (OpenAI, Cohere, Anthropic, Llama.cpp, Aphrodite)
- Unified interface for completion and chat requests
- Customizable sampling parameters
- Retry mechanism for generation attempts
- Flexible prompt handling and output processing

## Dependencies
This module requires the following external libraries:
- cohere
- openai
- anthropic

## Usage
To use this module, import the classes as follows:
```python
from llm_engine import LlmEngine, GenerationStep

# Initialize LlmEngine
engine = LlmEngine(mode="api", model="gpt-3.5-turbo")

# Create a GenerationStep instance
step = GenerationStep(prompt_path="path/to/prompt.txt", engine_wrapper=engine)

# Generate text
result = await step.generate(arguments)

