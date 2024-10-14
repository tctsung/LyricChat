from openai import OpenAI
import google.generativeai as genai
from pydantic import BaseModel, Field
from typing import List, Literal
from litellm import completion
import instructor
import os
class InstructorLLM:
    GEMINI_MODEL = "gemini-1.5-pro"
    OLLAMA_MODEL = "llama3"
    def __init__(self, deployment: Literal["cloud", "local"], GEMINI_API_KEY=None, base_url="http://localhost:11434/"):
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
            self.openAI_url = os.path.join(base_url, "v1/")   # make ollama endpoint became compatible with OpenAI
            self.model="ollama_chat/{model}".format(model = InstructorLLM.OLLAMA_MODEL)
            self.create_ollama()
        else:                                       # deployment == "cloud"
            self.GEMINI_API_KEY = GEMINI_API_KEY
            self.base_url = None                    # filler to use same code for LiteLLM completion
            self.model="gemini/{model}".format(model = InstructorLLM.GEMINI_MODEL)
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
        if self.GEMINI_API_KEY:   # set GEMINI_API_KEY if not None
            os.environ['GEMINI_API_KEY'] = self.GEMINI_API_KEY
        assert 'GEMINI_API_KEY' in os.environ, "Please set GEMINI_API_KEY as an environment variable"

        # create instructor client:
        model_name = "models/{model}".format(model=InstructorLLM.GEMINI_MODEL)
        self.client = instructor.from_gemini(
            client=genai.GenerativeModel(model_name=model_name),
            mode=instructor.Mode.GEMINI_JSON
            )

        # create 
    def run(self, messages, schema=None):
        """
        TODO: generate LLM response with/without Instructor structured output
              if schema
        Args:
            messages (List[Dict]): list of message dict
        
        """
        if schema:   # response without instructor
            return self.run_instructor(messages=messages, schema=schema) 
        else:                # instructor structured output
            return self.run_liteLLM(messages=messages)
    def run_instructor(self, messages, schema):
        """
        TODO: generate LLM response with Instructor structured output
        """
        # Conditionally add the model argument for local deployment:
        args = {"messages": messages,"response_model": schema}
        if self.deployment == "local": 
            args["model"] = InstructorLLM.OLLAMA_MODEL
        
        # get model response:
        response = self.client.chat.completions.create(**args)
        return response

    def run_liteLLM(self, messages):
        """
        TODO: generate LLM response with liteLLM
        """
        response = completion(
            model=self.model,   # recommend use ollama_chat then ollama
            messages=messages, 
            api_base=self.base_url
        )
        return response.choices[0].message.content
    
    def stream(self, messages):
        """
        TODO: stream LLM response with liteLLM
        Eg. 
        response = llm.stream(messages=[{ "content": "Hi!", "role": "user"}])
        for part in response:
            print(x)
        """
        response = completion(
            model=self.model,   # recommend use ollama_chat then ollama
            messages=messages, 
            api_base=self.base_url,
            stream=True
        )
        for part in response:
            yield (part.choices[0].delta.content or "")