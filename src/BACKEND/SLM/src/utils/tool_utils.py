import inspect
from typing import Any, Dict

class ToolSchemaValidator:
    @staticmethod
    def validate(tool_registry: Dict[str, Dict[str, Any]]) -> None:
        for name, tool in tool_registry.items():
            if 'function' not in tool:
                raise ValueError(f"Tool '{name}' missing 'function'")
            if not callable(tool['function']):
                raise TypeError(f"Tool '{name}' function is not callable")
            if 'parameters' not in tool:
                raise ValueError(f"Tool '{name}' missing 'parameters'")
            for param, spec in tool['parameters'].items():
                if 'type' not in spec or 'description' not in spec:
                    raise ValueError(f"Tool '{name}' parameter '{param}' missing type or description")

class RuntimeArgumentCoercer:
    TYPE_MAP = {
        'float': float,
        'int': int,
        'string': str,
        'bool': lambda x: str(x).lower() in ['true', '1', 'yes']
    }

    @staticmethod
    def coerce_args(params: Dict[str, Any], arg_spec: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        result = {}
        for key, spec in arg_spec.items():
            if key not in params:
                if spec.get("optional"):
                    result[key] = spec.get("default")
                    continue
                raise ValueError(f"Missing required parameter: {key}")
            expected_type = spec['type']
            try:
                result[key] = RuntimeArgumentCoercer.TYPE_MAP[expected_type](params[key])
            except Exception as e:
                raise TypeError(f"Failed to coerce '{key}' to {expected_type}: {e}")
        return result

class ToolRunner:
    @staticmethod
    def run(tool_def: Dict[str, Any], args: Dict[str, Any]) -> Any:
        coerced_args = RuntimeArgumentCoercer.coerce_args(args, tool_def['parameters'])
        return tool_def['function'](**coerced_args)

class JSONSchemaExporter:
    @staticmethod
    def export(tool_name: str, tool_def: Dict[str, Any]) -> Dict[str, Any]:
        schema = {
            "name": tool_name,
            "description": tool_def.get("function").__doc__ or "",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
        for param, spec in tool_def['parameters'].items():
            schema["parameters"]["properties"][param] = {
                "type": spec['type'],
                "description": spec['description']
            }
            if not spec.get("optional"):
                schema["parameters"]["required"].append(param)
        return schema