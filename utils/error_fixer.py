from typing import Dict

from openai.types.shared_params.function_definition import FunctionDefinition


def mock(
    error_info: Exception, tool_name: str, tool_arguments: Dict, tool_info: FunctionDefinition
):
    # tool_name is True, tool_arguments may be Wrong, you need to fix it according to tool_info and error_info
    # your mock logic
    fix_input_dict = {}
    # if not resolve, return {}
    return fix_input_dict