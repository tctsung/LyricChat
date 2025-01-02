from openai import OpenAI
import google.generativeai as genai
from pydantic import BaseModel, Field
from typing import List, Literal, Annotated
from typing_extensions import TypedDict
from litellm import completion
import instructor
import os
from operator import add


######### helper functions that are similar to LangChain but more flxible & works with class InstructorLLM ####
class MsgState(TypedDict):
    messages: Annotated[list[dict], add]  # list + list


# Replacements of HumanMessage, AIMessage, SystemMessage in langchain_core.messages
human_msg = lambda content: {"role": "user", "content": content}
AI_msg = lambda content: {"role": "assistant", "content": content}
sys_msg = lambda content: {"role": "system", "content": content}
lyric_msg = lambda lyric: {
    "role": "user",
    "content": f"<input_lyrics> {lyric} </input_lyrics>",
}


class InstructorLLM:
    GEMINI_MODEL = "gemini-1.5-pro"  # "gemini-1.5-pro"
    OLLAMA_MODEL = "llama3"

    def __init__(
        self,
        deployment: Literal["cloud", "local"],
        GEMINI_API_KEY=None,
        base_url="http://localhost:11434/",
    ):
        """
        TODO: generate a friendly LLM response interface compatible with Instructor structured output. Use stream/run to get response.
        Args:
            model (str): Currently only support cloud (gemini-1.5-pro at backend) or local (ollama/llama3 at backend)
            GEMINI_API_KEY (str): set GEMINI_API_KEY if not in environment & deployment == "cloud"
            base_url (str): local url for ollama
        """
        self.deployment = deployment

        if deployment == "local":
            self.base_url = base_url
            self.openAI_url = os.path.join(
                base_url, "v1/"
            )  # make ollama endpoint became compatible with OpenAI
            self.model = "ollama_chat/{model}".format(model=InstructorLLM.OLLAMA_MODEL)
            self.create_ollama()
        else:  # deployment == "cloud"
            self.GEMINI_API_KEY = GEMINI_API_KEY
            self.base_url = None  # filler to use same code for LiteLLM completion
            self.model = "gemini/{model}".format(model=InstructorLLM.GEMINI_MODEL)
            self.create_gemini()

    def create_ollama(self):
        """
        TODO: create Instructor client for local ollama model
        """
        self.client = instructor.from_openai(
            OpenAI(
                base_url=self.openAI_url,
                api_key="ollama",  # required, but unused
            ),
            mode=instructor.Mode.JSON,
        )

    def create_gemini(self):
        """
        TODO: create Instructor client for gemini model
        """
        if self.GEMINI_API_KEY:  # set GEMINI_API_KEY if not None
            os.environ["GEMINI_API_KEY"] = self.GEMINI_API_KEY
        assert (
            "GEMINI_API_KEY" in os.environ
        ), "Please set GEMINI_API_KEY as an environment variable or pass it as an argument"

        # set API key
        genai.configure(api_key=self.GEMINI_API_KEY)

        # create instructor client:
        model_name = "models/{model}".format(model=InstructorLLM.GEMINI_MODEL)
        self.client = instructor.from_gemini(
            client=genai.GenerativeModel(model_name=model_name),
            mode=instructor.Mode.GEMINI_JSON,
        )

        # create

    def run(self, messages, schema=None, max_retries=3):
        """
        TODO: generate LLM response with/without Instructor structured output
              if schema
        Args:
            messages (List[Dict]): list of message dict

        """
        if isinstance(messages, str):
            messages = [human_msg(messages)]
        assert isinstance(messages, list), "messages must be a list"
        if schema:  # response without instructor
            return self._run_instructor(
                messages=messages, schema=schema, max_retries=max_retries
            )
        else:  # instructor structured output
            return self._run_liteLLM(messages=messages)

    def _run_instructor(self, messages, schema, max_retries):
        """
        TODO: generate LLM response with Instructor structured output
        """
        # Conditionally add the model argument for local deployment:
        args = {
            "messages": messages,
            "response_model": schema,
            "max_retries": max_retries,
        }
        if self.deployment == "local":
            args["model"] = InstructorLLM.OLLAMA_MODEL

        # get model response:
        response = self.client.chat.completions.create(**args)
        return response

    def _run_liteLLM(self, messages):
        """
        TODO: generate LLM response with liteLLM
        """
        response = completion(
            model=self.model,  # recommend use ollama_chat then ollama
            messages=messages,
            api_base=self.base_url,
            # temperature=temperature
        )
        return response.choices[0].message.content or ""

    def stream(self, messages):
        """
        TODO: stream LLM response with liteLLM
        Eg.
        response = llm.stream(messages=[{ "content": "Hi!", "role": "user"}])
        for part in response:
            print(x)
        """
        response = completion(
            model=self.model,  # recommend use ollama_chat then ollama
            messages=messages,
            api_base=self.base_url,
            stream=True,
        )
        for part in response:
            yield (part.choices[0].delta.content or "")
