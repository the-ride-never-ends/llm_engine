"""
LlmEngine is a class for interacting with various Language Model APIs and engines.

This class provides a unified interface for submitting completion and chat requests
to different LLM backends, including OpenAI API, Cohere, Anthropic, Llama.cpp, and Aphrodite.

Attributes:
    mode (str): The mode of operation ('api', 'aphrodite', 'llama.cpp', 'cohere', or 'anthropic').
    model (str): The name of the model to use.
    engine (AsyncAphrodite): The Aphrodite engine instance (only for 'aphrodite' mode).
    client (AsyncOpenAI or AsyncClient): The client instance for API calls.
    defaults (dict): Default parameters for sampling.

Methods:
    add_defaults_if_absent(sampling_params: dict) -> dict:
        Adds default sampling parameters if they are not present in the input.

    async submit_completion(prompt: str, sampling_params: dict) -> Union[str, Tuple[str, bool]]:
        Submits a completion request to the LLM engine.

    async submit_chat(messages: List[dict], sampling_params: dict) -> Tuple[str, bool]:
        Submits a chat request to the LLM engine.
"""


import asyncio
import platform
import sys
import uuid

from openai import AsyncOpenAI
import cohere
import anthropic


from utils.llm_engine.make_llama_cpp_async_api_call import (
    make_llama_cpp_async_api_call,
)

from ..config.config import *
from ..logger.logger import Logger
logger = Logger(logger_name=__name__)


if platform.system() == "Linux":
    try: # Try to import the aphrodite if we're on Linux.
        from aphrodite import (
            EngineArgs,
            AphroditeEngine,
            SamplingParams,
            AsyncAphrodite,
            AsyncEngineArgs,
        )
    except ImportError:
        import subprocess
        import sys

        logger.info("Aphrodite package not found. Attempting to install...")
        try: # If the import fails, try to install it, then import it.
            subprocess.check_call([sys.executable, "-m", "pip", "install", "aphrodite"])
            logger.info("Aphrodite package installed successfully.")
            from aphrodite import (
                EngineArgs,
                AphroditeEngine,
                SamplingParams,
                AsyncAphrodite,
                AsyncEngineArgs,
            )
        except Exception as e:
            logger.warning(f"Aphrodite package failed to install: {e}\nStick to Llama CPP or API modes")
else:
    logger.info("Aphrodite cannot be installed on a non-Linux system.\nStick to Llama CPP or API modes")


def make_id():
    return str(uuid.uuid4())


class LlmEngine:
    def __init__(
        self,
        model: str,
        api_key: str=None,
        base_url: str=None,
        mode: str="api",  # can be one of api, aphrodite, llama.cpp, cohere, or anthropic
        quantization: str="gptq",  # only needed if using aphrodite mode
    ):
        self.mode = mode
        self.model = model
        if mode == "aphrodite":
            engine_args = AsyncEngineArgs(
                model=model,
                quantization=quantization,
                engine_use_ray=False,
                disable_log_requests=True,
                max_model_len=12000,
                dtype="float16",
            )
            self.engine = AsyncAphrodite.from_engine_args(engine_args)
        if mode == "cohere":
            self.client = cohere.AsyncClient(api_key=api_key)
        elif mode == "api":
            self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.defaults = {
            "temperature": 1,
            "top_p": 1,
            "max_tokens": 3000,
            "stop": []
        }

    def add_defaults_if_absent(self, sampling_params: dict) -> dict:
        for param, value in self.defaults.items():
            # If the parameter is not in the sampling params, add it with the default value
            sampling_params.setdefault(param, value)
        return sampling_params

    async def submit_completion(
        self, prompt, sampling_params
    ):  # Submit request and wait for it to stream back fully
        sampling_params = self.add_defaults_if_absent(sampling_params)
        if "n_predict" not in sampling_params and self.mode == "llamacpp":
            sampling_params["n_predict"] = sampling_params["max_tokens"]

        match self.mode:
            case "llamacpp":
                return await make_llama_cpp_async_api_call(
                    prompt=prompt, sampling_parameters=sampling_params
                )

            case "aphrodite":
                aphrodite_sampling_params = SamplingParams(**sampling_params)
                request_id = make_id()
                outputs = []
                final_output = None
                async for request_output in self.engine.generate(
                    prompt, aphrodite_sampling_params, request_id
                ):
                    outputs.append(request_output.outputs[0].text)
                    final_output = request_output

                return final_output.prompt + final_output.outputs[0].text

            case "api":
                timed_out = False
                completion = ""
                stream = await self.client.completions.create(
                    model=self.model,
                    prompt=prompt,
                    temperature=sampling_params["temperature"],
                    top_p=sampling_params["top_p"],
                    stop=sampling_params["stop"],
                    max_tokens=sampling_params["max_tokens"],
                    stream=True,
                    timeout=360,
                )
                async for chunk in stream:
                    try:
                        completion = completion + chunk.choices[0].delta.content
                    except:
                        timed_out = True

                return prompt + completion, timed_out

            case "cohere":
                logger.error("Cohere not compatible with completion mode!")
                raise Exception("Cohere not compatible with completion mode!")

            case _:
                logger.error(f"Completion mode not supported for {self.mode}!")
                raise Exception(f"Completion mode not supported for {self.mode}!")


    async def submit_chat(
        self, messages, sampling_params
    ):  # Submit request and wait for it to stream back fully
        sampling_params = self.add_defaults_if_absent(sampling_params)

        match self.mode:
            case "llamacpp":
                return await make_llama_cpp_async_api_call(
                    messages=messages, sampling_parameters=sampling_params
                )

            case "cohere":
                timed_out = False
                completion = ""
                messages_cohereified = [
                    {
                        "role": "USER" if message["role"] == "user" else "CHATBOT",
                        "message": message["content"],
                    }
                    for message in messages
                ]
                stream = self.client.chat_stream(
                    model=self.model,
                    chat_history=messages_cohereified[1:-1],
                    message=messages_cohereified[-1]["message"],
                    preamble=messages_cohereified[0]["message"],
                    temperature=sampling_params["temperature"],
                    p=sampling_params["top_p"],
                    stop_sequences=sampling_params["stop"],
                    max_tokens=sampling_params["max_tokens"],
                )
                async for chunk in stream:
                    try:
                        if chunk.event_type == "text-generation":
                            completion = completion + chunk.text
                    except Exception as e:
                        logger.exception("THIS RESPONSE TIMED OUT PARTWAY THROUGH GENERATION!\n{e}\ntimed_out = True",f=True)
                        timed_out = True

                return completion, timed_out

            case "api":
                completion = ""
                timed_out = False
                stream = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=sampling_params["temperature"],
                    top_p=sampling_params["top_p"],
                    stop=sampling_params["stop"],
                    max_tokens=sampling_params["max_tokens"],
                    stream=True,
                )
                async for chunk in stream:
                    try:
                        if chunk.choices[0].delta.content:
                            completion = completion + chunk.choices[0].delta.content
                    except Exception as e:
                        logger.exception("\n\n------------CAUGHT EXCEPTION DURING GENERATION\n{e}\ntimed_out = True\n\n-----/\------",f=True)
                        timed_out = True

                return completion, timed_out

            case "aphrodite":
                logger.error("Aphrodite not compatible with chat mode!")
                raise Exception("Aphrodite not compatible with chat mode!")

            case _:
                logger.error(f"Chat mode not supported for {self.mode}!")
                raise Exception(f"Chat mode not supported for {self.mode}!")
