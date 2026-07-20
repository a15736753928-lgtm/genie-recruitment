---
name: position-creation-id-generation-bug
description: 创建岗位时服务端出现数据库ID字段未自动生成错误
metadata:
  type: business_rule
  scope: recruitment
created: '2026-07-20T07:22:55.767098+00:00'
updated: '2026-07-20T07:22:55.767098+00:00'
---

创建名为'测试岗位'时失败，报数据库错误，提示ID字段未自动生成。疑似positions表的ID默认生成逻辑存在缺陷，需联系开发团队检查并修复。临时方案可尝试手动指定ID。Why: 避免后续岗位创建失败影响招聘流程。How to apply: 修复前创建岗位时需注意此Bug，修复后回归验证。
