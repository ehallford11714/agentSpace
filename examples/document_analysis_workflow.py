import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from workflow.tasks import Task
from workflow.workflows import Workflow
from toolLib.tool_registry import ToolRegistry
from toolLib.file_tools import FileTool

# Example tasks using FileTool
class CreateFileTask(Task):
    def execute(self, tool_registry: ToolRegistry):
        file_tool = tool_registry.get_tool_instance(self.tool)
        return file_tool.create_file(**self.parameters)

class ReadFileTask(Task):
    def execute(self, tool_registry: ToolRegistry):
        file_tool = tool_registry.get_tool_instance(self.tool)
        return file_tool.read_file(**self.parameters)

class DeleteFileTask(Task):
    def execute(self, tool_registry: ToolRegistry):
        file_tool = tool_registry.get_tool_instance(self.tool)
        return file_tool.delete_file(**self.parameters)


def main():
    registry = ToolRegistry()
    registry.register_tool('file', FileTool)
    registry.create_tool_instance('file', {'base_path': '.'})

    wf = Workflow('demo', tool_registry=registry)
    wf.add_task(CreateFileTask('create', 'file', {'path': 'demo.txt', 'content': 'hello world'}))
    wf.add_task(ReadFileTask('read', 'file', {'path': 'demo.txt'}, dependencies=['create']))
    wf.add_task(DeleteFileTask('delete', 'file', {'path': 'demo.txt'}, dependencies=['read']))

    result = wf.execute()
    print(result)

if __name__ == '__main__':
    main()
