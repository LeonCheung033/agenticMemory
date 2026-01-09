"""测试工具基类."""
import pytest
from src.agemem.tools.base import BaseTool, ToolSchema


class ConcreteTool(BaseTool):
    """用于测试的具体工具实现."""

    def __init__(self):
        super().__init__()
        self._schema = ToolSchema(
            name="test_tool",
            description="A test tool",
            parameters={
                "type": "object",
                "properties": {
                    "param1": {"type": "string", "description": "First parameter"},
                    "param2": {"type": "integer", "description": "Second parameter"},
                },
                "required": ["param1"],
            },
        )

    @property
    def schema(self) -> ToolSchema:
        return self._schema

    def execute(self, **kwargs) -> dict:
        """执行工具."""
        return {"success": True, "result": {"message": "Tool executed"}}


class TestToolSchema:
    """测试ToolSchema."""

    def test_create_schema(self):
        """测试创建Schema."""
        schema = ToolSchema(
            name="test_tool",
            description="Test description",
            parameters={"type": "object", "properties": {}},
        )

        assert schema.name == "test_tool"
        assert schema.description == "Test description"
        assert schema.parameters == {"type": "object", "properties": {}}

    def test_to_dict(self):
        """测试转换为字典."""
        schema = ToolSchema(
            name="test_tool",
            description="Test description",
            parameters={"type": "object", "properties": {}},
        )

        data = schema.to_dict()
        assert data["name"] == "test_tool"
        assert data["description"] == "Test description"
        assert data["parameters"] == {"type": "object", "properties": {}}


class TestBaseTool:
    """测试BaseTool."""

    def test_base_tool_is_abstract(self):
        """测试BaseTool是抽象类，不能直接实例化."""
        with pytest.raises(TypeError):
            BaseTool()

    def test_concrete_tool(self):
        """测试具体工具实现."""
        tool = ConcreteTool()
        assert tool.schema.name == "test_tool"
        assert tool.schema.description == "A test tool"

    def test_execute(self):
        """测试执行工具."""
        tool = ConcreteTool()
        result = tool.execute(param1="value1", param2=42)

        assert result["success"] is True
        assert "result" in result

    def test_validate_args_valid(self):
        """测试参数验证 - 有效参数."""
        tool = ConcreteTool()
        args = {"param1": "value1", "param2": 42}

        is_valid, error = tool.validate_args(args)
        assert is_valid is True
        assert error is None

    def test_validate_args_missing_required(self):
        """测试参数验证 - 缺少必需参数."""
        tool = ConcreteTool()
        args = {"param2": 42}  # 缺少param1

        is_valid, error = tool.validate_args(args)
        assert is_valid is False
        assert "Missing required parameter: param1" in error

    def test_validate_args_extra_params(self):
        """测试参数验证 - 额外参数（应该允许）."""
        tool = ConcreteTool()
        args = {"param1": "value1", "param2": 42, "extra": "value"}

        is_valid, error = tool.validate_args(args)
        # 额外参数应该被允许
        assert is_valid is True

    def test_validate_args_empty_required(self):
        """测试参数验证 - 没有必需参数的情况."""
        schema = ToolSchema(
            name="no_required",
            description="No required params",
            parameters={"type": "object", "properties": {}, "required": []},
        )

        class NoRequiredTool(BaseTool):
            def __init__(self):
                super().__init__()
                self._schema = schema

            @property
            def schema(self) -> ToolSchema:
                return self._schema

            def execute(self, **kwargs) -> dict:
                return {"success": True}

        tool = NoRequiredTool()
        is_valid, error = tool.validate_args({})
        assert is_valid is True
