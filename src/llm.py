from openai import OpenAI
import google.generativeai as genai
from pydantic import BaseModel, Field
from typing import List, Literal, Annotated
from typing_extensions import TypedDict
from litellm import completion
import instructor
import os

# Replacements of HumanMessage, AIMessage, SystemMessage in langchain_core.messages
human_msg = lambda content: {"role": "user", "content": content}
AI_msg = lambda content: {"role": "assistant", "content": content}
sys_msg = lambda content: {"role": "system", "content": content}
lyric_msg = lambda lyric: {
    "role": "user",
    "content": f"<input_lyrics> {lyric} </input_lyrics>",
}


class InstructorLLM:
    GEMINI_MODEL = "gemini-1.5-flash"  # "gemini-1.5-pro"
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

    def run(
        self,
        messages,
        schema=None,
        max_retries=3,
        chat_history: List[dict] = [],
        memory: int = 2,
    ):
        """
        TODO: generate LLM response with/without Instructor structured output
              if schema
        Args:
            messages (List[Dict]): list of message dict
            schema (Dict): schema for instructor structured output
            max_retries (int): number of retries
            chat_history (List[Dict]): list of chat history
            memory (int): number of memory to include in the messages
        """
        messages = msg_w_memory(messages, chat_history, memory)

        if schema:  # instructor structured output
            return self._run_instructor(
                messages=messages, schema=schema, max_retries=max_retries
            )
        else:  # response without instructor
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

    def stream(self, messages, chat_history: List[dict] = [], memory: int = 2):
        """
        TODO: stream LLM response with liteLLM
        Args:
            messages (List[Dict]): list of message dict
            yield_response (bool): if True, yield response in chunks; otherwise return a generator
        Eg. 1
        response = llm.stream(messages="why is the sky blue?", yield_response=True)
        st.write_stream(response)
        Eg. 2
        response = llm.stream(messages="why is the sky blue?", yield_response=False)
        for chunk in response:
            print(chunk.choices[0].delta.content or "")
        """
        messages = msg_w_memory(messages, chat_history, memory)
        response = completion(
            model=self.model,  # recommend use ollama_chat then ollama
            messages=messages,
            api_base=self.base_url,
            stream=True,
        )
        return response


###### Helper functions ######
def msg_w_memory(messages, chat_history, memory):
    """TODO: prepare messages with memory"""
    if isinstance(messages, str):
        messages = [human_msg(messages)]
    assert isinstance(messages, list), "messages must be a list"
    assert isinstance(memory, int), "memory must be an integer"

    # Extract system message if present
    system_message = [msg for msg in messages if msg["role"] == "system"]
    if system_message:  # remove system message from messages
        messages = [msg for msg in messages if msg["role"] != "system"]

    # Limit chat history based on memory
    if memory > 0:
        chat_history = chat_history[-(2 * memory) :]

    # Combine system message, chat history, and new messages
    combined_messages = []
    combined_messages.extend(system_message)
    combined_messages.extend(chat_history)
    combined_messages.extend(messages)
    return combined_messages


def preprocess_stream(chunk):
    """
    TODO: helper for InstructorLLM.stream to preprocess the chunk
    """
    return chunk.choices[0].delta.content or ""
