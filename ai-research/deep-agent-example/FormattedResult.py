import json
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from datetime import datetime


@dataclass
class FormattedResult:
    """格式化后的结果"""
    summary: str
    messages: List[Dict]
    token_usage: Dict[str, int]
    tools_used: List[str]
    duration_ms: Optional[int] = None


class ResultFormatter:
    """Agent 结果格式化器"""

    def __init__(self, max_content_length: int = 1000):
        self.max_length = max_content_length

    def format(self, result: Dict, start_time: Optional[float] = None) -> FormattedResult:
        """格式化 invoke 结果"""

        messages = self._extract_messages(result)
        token_usage = self._calculate_tokens(messages)
        tools_used = self._extract_tools(messages)

        # 计算耗时
        duration = None
        if start_time:
            duration = int((datetime.now().timestamp() - start_time) * 1000)

        # 生成摘要
        summary = self._generate_summary(messages)

        return FormattedResult(
            summary=summary,
            messages=messages,
            token_usage=token_usage,
            tools_used=tools_used,
            duration_ms=duration
        )

    def _extract_messages(self, result: Dict) -> List[Dict]:
        """提取消息列表"""
        messages = []
        raw_messages = result.get("messages", [])

        for msg in raw_messages:
            messages.append({
                "role": getattr(msg, "type", "unknown"),
                "content": getattr(msg, "content", str(msg))[:self.max_length],
                "tool_calls": getattr(msg, "tool_calls", None),
                "usage": getattr(msg, "usage_metadata", {})
            })

        return messages

    def _calculate_tokens(self, messages: List[Dict]) -> Dict[str, int]:
        """计算 Token 使用"""
        total_input = sum(m["usage"].get("input_tokens", 0) for m in messages)
        total_output = sum(m["usage"].get("output_tokens", 0) for m in messages)

        return {
            "input": total_input,
            "output": total_output,
            "total": total_input + total_output
        }

    def _extract_tools(self, messages: List[Dict]) -> List[str]:
        """提取使用的工具"""
        tools = []
        for msg in messages:
            if msg["tool_calls"]:
                for tc in msg["tool_calls"]:
                    tools.append(tc.get("name", "unknown"))
        return list(set(tools))  # 去重

    def _generate_summary(self, messages: List[Dict]) -> str:
        """生成摘要"""
        if not messages:
            return "无消息"

        last_msg = messages[-1]
        content = last_msg["content"]

        # 截断摘要
        return content[:200] + "..." if len(content) > 200 else content

    def print(self, result: FormattedResult):
        """打印格式化结果"""
        print("=" * 70)
        print(f"🤖 Agent 执行结果")
        print("=" * 70)

        # 摘要
        print(f"\n📋 摘要: {result.summary}")

        # 性能
        # if result.duration_ms:
        print(f"⏱️  耗时: {result.duration_ms}ms")

        # Token
        print(f"\n📊 Token 使用:")
        print(f"   输入: {result.token_usage['input']}")
        print(f"   输出: {result.token_usage['output']}")
        print(f"   总计: {result.token_usage['total']}")

        # 工具
        if result.tools_used:
            print(f"\n🔧 使用工具: {', '.join(result.tools_used)}")

        # 消息详情
        print(f"\n📨 消息记录 ({len(result.messages)} 条):")
        for i, msg in enumerate(result.messages, 1):
            role_icon = {"human": "👤", "ai": "🤖", "tool": "🔧"}.get(msg["role"], "❓")
            print(f"\n   {role_icon} [{i}] {msg['role'].upper()}")
            print(f"      {msg['content'][:300]}...")

        print("\n" + "=" * 70)


