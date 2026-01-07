from BaseAgent import BaseAgent
from typing import List, Dict, Any, Optional, Tuple
import json
import sys
import logging
class BaseWorkFlow:
    def __init__(self):
        self.agents = {}
        self.agents.update({"CaptionAgent": BaseAgent(name="CaptionAgent", system_prompt=f"You are a Caption Agent that can assign subtasks to the SearchAgent and SummarizeAgent to complete the task, and submit next agent of the final result. Follow the format of <submit>{{'Agent':'NextAgentName','Instruction':'Instruction to the next agent'}}</submit> to select the next agent. And <submit>FinalResult:(FinalResult)</submit> to submit the final result.", tools=["google-search","fetch"])})
        self.agents.update({"SearchAgent":BaseAgent(name="SearchAgent", system_prompt="You are a Search Agent that can use google search to find information, and submit valuable information and potential helpful links", tools=["google-search"])})
        self.agents.update({"SummarizeAgent":BaseAgent(name="SummarizeAgent", system_prompt="You are a Web Content Summarize Agent that can use fetch to get web content of a given url and submit the summary of the web content", tools=["fetch"])})
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('[%(asctime)s] %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.max_turns = 10
    def run_workflow(self, input):
        current_agent = self.agents["CaptionAgent"]
        for i in range(self.max_turns):
            response = current_agent.run(input)
            print("Step",i,f"Response: {response}")
            if "<submit>" in response:

                self.logger.info(f"Submit result: {response}")
                submit_result = response.split("<submit>")[1].split("</submit>")[0]
                if "FinalResult:" in submit_result or "FinalResult:" in response:
                    submit_result = json.loads(submit_result)
                    final_result = submit_result["FinalResult"]
                    self.logger.info(f"Final result: {final_result}")
                    return final_result
                elif "Agent" in submit_result and "Instruction" in submit_result:
                    #print(submit_result)
                    submit_result = json.loads(submit_result)
                    next_agent = submit_result["Agent"]
                    instruction = submit_result["Instruction"]
                    self.logger.info(f"Next agent: {next_agent}, Instruction: {instruction}")
                    input = instruction
                    response = self.agents[next_agent].run(input)
                    self.logger.info(f"Agent {next_agent} result: {response}")
                    submit_result = response.split("<submit>")[1].split("</submit>")[0]
                    input = f"The response form {self.agents[next_agent].name} is {submit_result}"
                    #current_agent = self.agents[next_agent]
                


if __name__ == "__main__":
    workflow = BaseWorkFlow()
    workflow.run_workflow("Which program funded the SUPPLY project involving EHA, when did the project start, and how long will it last?")