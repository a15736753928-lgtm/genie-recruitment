"""LLM Provider Adapter 层。

每个 provider type 对应一个 adapter，封装：
- 原生 SDK 客户端创建
- LangChain 兼容对象（ChatModel）
- 连通性测试
"""
