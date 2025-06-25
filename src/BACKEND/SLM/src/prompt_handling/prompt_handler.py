import json
from pathlib import Path
from typing import List, Optional, Union

from ..models.base_models import Tool

class PromptHandlerError(Exception):
    """Base exception for PromptHandler errors."""
    pass

class PromptHandler:
    """Handles prompt construction and tool management."""
    
    def __init__(
        self, 
        tools_path: Union[str, Path],
        system_behavior: Optional[str] = None
    ):
        """
        Initialize the PromptHandler.
        
        Args:
            tools_path: Path to the JSON file containing tool definitions
            system_behavior: Optional system behavior string
        """
        self.tools_path = Path(tools_path)
        self.system_behavior = system_behavior or (
            "You are a highly capable and versatile language model. "
            "You can understand and generate human-like text, reason through "
            "complex problems, and assist with a wide range of tasks."
        )
        self.tools = self._load_tools()

    def _load_tools(self) -> List[Tool]:
        """
        Load tools from the JSON file.
        
        Raises:
            PromptHandlerError: If the file cannot be read or parsed
        """
        try:
            with open(self.tools_path, 'r') as f:
                data = json.load(f)
            if not isinstance(data, dict) or 'tools' not in data:
                raise PromptHandlerError("Invalid tools file format. Expected JSON object with 'tools' array.")
            return [Tool(**tool_data) for tool_data in data['tools']]
        except json.JSONDecodeError as e:
            raise PromptHandlerError(f"Failed to parse tools file: {e}")
        except OSError as e:
            raise PromptHandlerError(f"Failed to read tools file: {e}")
        except Exception as e:
            raise PromptHandlerError(f"Error loading tools: {e}")

    def get_tool_json(self) -> str:
        """Convert tools to a compact JSON string."""
        tool_dicts = [
            json.loads(tool.json(exclude_none=True))
            for tool in self.tools
        ]
        return json.dumps(tool_dicts, separators=(',', ':'))

    def construct_prompt(self, user_query: str) -> str:
        """
        Construct a prompt with system behavior, tools, and user query.
        
        Args:
            user_query: The user's input query
            
        Returns:
            A formatted prompt string
        """
        tool_json = json.dumps(
            json.loads(self.get_tool_json()),
            indent=4
        )
        
        return (
            f"<|system|>\n\t{self.system_behavior}\n"
            f"<|tool|>\n\t{tool_json}\n<|/tool|>\n<|end|>\n"
            f"<|user|>\n\t{user_query}\n<|end|>\n<|assistant|>"
        )