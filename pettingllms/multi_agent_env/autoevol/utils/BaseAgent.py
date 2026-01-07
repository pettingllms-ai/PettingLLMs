from BaseOpenAI import AIClient
import os
from typing import List, Dict, Any, Optional, Tuple
import json
import sys
#sys.path.append("/mnt/afs/zhangyaolun/safe_model/masrl/utils/environments")
from environments.search_env import SearchEnvironment
import logging
import ast
class BaseAgent:
    def __init__(self, name: str, system_prompt: str, tools: List[Dict[str, Any]]):
        self.name = name
        self.system_prompt = system_prompt
        self.tools = tools
        self.max_turns = 10
        api_base = os.getenv("API_BASE", "https://api.openai.com/v1")
        api_key = os.getenv("API_KEY")
        chat_model = os.getenv("CHAT_MODEL","gpt-4o-mini")
        max_answer_tokens = os.getenv("MAX_ANSWER_TOKENS", 8192)
        self.environment = os.getenv("ENVIRONMENT", "search")
        self.ai_client = AIClient(api_base=api_base, api_key=api_key, chat_model=chat_model, max_answer_tokens=max_answer_tokens)
        self.system_prompt = self._prepare_system_prompt()
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('[%(asctime)s] %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.memory = None
    def _prepare_tools(self):
        if self.environment == "search":
            self.environment = SearchEnvironment(serper_key=os.getenv("SERPER_KEY"))
        tools = []
        if "google-search" in self.tools:
            tools.append({
                "type": "function",
                "function": {
                    "name": "google-search",
                    "description": "Use Google Search to find information",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The query to search for"
                            }
                        },
                        "required": ["query"]
                    }
                }
            })
        if "fetch_data" in self.tools:
            tools.append({
                "type": "function",
                "function": {
                    "name": "fetch_data",
                    "description": "Fetch data from a given url",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "url": {
                                "type": "string",
                                "description": "The url to fetch data from"
                            }
                        },
                        "required": ["url"]
                    }
                }
            })
        return tools
    def _prepare_system_prompt(self):
        #if "google-search" in self.tools:
        #    self.system_prompt + f"\n\nYou can use google search to get the information you need. Follow the json format: {{'tool_name': 'google-search', 'parameters': '{'query': 'your query'}'}}"
        #if "fetch_data" in self.tools:
        #    self.system_prompt + f"\n\nYou can use fetch to get web content of a given url. Follow the json format: {{'tool_name': 'fetch', 'parameters': '{'url': 'your url'}'}}"
        tools = self._prepare_tools()
        self.system_prompt = (
            self.system_prompt
            + f"\n\nYou have the following tools: {tools}.\n"
            "When you need to call a tool, output it in the following format:\n"
            "<tool_call>{\n"
            '  "name": "tool_name",\n'
            '  "parameters": {\n'
            '    "parameter_name": "parameter_value"\n'
            "  }\n"
            "}</tool_call>\n"
            "When you finish the task, output the final result between <submit>...</submit>."
        )
        return self.system_prompt


    
    def set_memory(self, memory:List[Dict[str, Any]]):
        self.memory = memory
    def run(self, input):
        if self.memory is not None:
            message = self.memory
        else:
            message = [{"role": "system", "content": self.system_prompt},{"role": "user", "content": input}]
        tokens = 0
        for i in range(self.max_turns):
            response, prompt_tokens, completion_tokens = self.ai_client.chat(message)
            #self.logger.info(f"Agent {self.name} response: {response}")
            tokens += prompt_tokens + completion_tokens
            message.append({"role": "assistant", "content": response})
            if "<tool_call>" in response:
                tool_call = json.loads(response.split("<tool_call>")[1].split("</tool_call>")[0])
                tool_name = tool_call["name"]
                tool_parameters = tool_call["parameters"]
                if tool_name == "google-search":
                    tool_response = self.environment.search(tool_parameters["query"])
                elif tool_name == "fetch_data":
                    tool_response = self.environment.fetch(tool_parameters["url"])
                message.append({"role": "user", "content": f"The result of the {tool_name} tool call is: {tool_response}"})
            if "<submit>" in response:
                #print(f"Agent {self.name} submit result: {response}")
                #submit_result = json.loads(response.split("<submit>")[1].split("</submit>")[0])
                

                raw = response.split("<submit>")[1].split("</submit>")[0].strip()
                self.set_memory(message)
                return raw
            if i == self.max_turns - 1:
                message.append({"role": "user", "content": "Reach the max turns, please submit the result. Follow <submit>...</submit> to submit the detailed result."})
        self.set_memory(message)
        
        return response

    