from typing import Iterable, Optional

import rich
from openai import OpenAI
from openai.types.chat import ChatCompletion, ChatCompletionToolParam, ChatCompletionMessageParam
import httpx

from config import LLMConfig, GenerationOptions as LLMGenerationOptions


def sync_request_llm(
    llm_config: LLMConfig,
    messages: Iterable[ChatCompletionMessageParam],
    tools: Optional[Iterable[ChatCompletionToolParam]] = None,
    generation_config: LLMGenerationOptions = LLMGenerationOptions()
) -> ChatCompletion:
    api_key = llm_config.api_key
    base_url = llm_config.base_url
    model_name = llm_config.model

    http_client = httpx.Client(verify=False)
    client = OpenAI(api_key=api_key, base_url=base_url, http_client=http_client)
    completion = client.chat.completions.create(
        model=model_name,
        messages=messages,
        stream=generation_config.stream,
        temperature=generation_config.temperature,
        tools=tools,
        extra_body=generation_config.extra_body,
        timeout=generation_config.timeout,
        max_tokens=generation_config.max_tokens,
        top_p=generation_config.top_p,
        presence_penalty=generation_config.presence_penalty
    )  # type: ignore
    return completion


if __name__ == "__main__":

    result = sync_request_llm(
        llm_config=LLMConfig(
            api_key="tokenabc123",
            base_url="http://127.0.0.1:12345/v1",
            model="/data1/yaoys6/models/Qwen3-32B"
            #vllm serve "/data1/yaoys6/models/Qwen3-32B" --served-model-name qwen-32b --port 12345 --dtype auto --api-key "tokenabc123" --enable-reasoning --reasoning-parser deepseek_r1 --tensor-parallel-size 2
        ),
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": "What is the meaning of life?"
            }
        ],
        generation_config=LLMGenerationOptions(
            temperature=0.2,
            timeout=120,
        )
    )
    rich.print(result)