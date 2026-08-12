## Prompt 1
你是一个专业的简历解析器。请从简历文本中提取个人基本信息和经历，返回纯JSON。
[[position_hint]]

【简历文本】
[[text_12000]]

返回如下JSON（找不到的填"未知"，应届经验填"应届"）：
{
    "name": "姓名",
    "gender": "男/女/未知",
    "age": null,
    "birthDate": "出生日期如2003-12-12，无则null",
    "education": "最高学历：大专/本科/硕士/博士/其他/未知",
    "experience": "工作年限如7年；无工作经历填在校中",
    "phone": "手机号，未知填未知",
    "email": "邮箱，未知填未知",
    "ethnicity": "民族，未提及填汉族",
    "nativePlace": "籍贯或现居地，未知填未知",
    "skills": ["技能1", "技能2"],
    "educationHistory": [{"school": "学校", "degree": "学位", "major": "专业", "period": "时间段"}],
    "workHistory": [{"company": "公司", "role": "职位", "period": "时间段", "description": "工作描述"}],
    "projectHistory": [{"name": "项目名", "role": "角色", "period": "时间段", "description": "项目描述"}]
}

只返回JSON。严格要求：只能提取原文明确出现的信息，不得臆造。age留null，识别到出生日期填birthDate。

## Prompt 2
从以下简历文本中提取6项特定信息，返回纯JSON。

【简历文本】
[[text_12000]]

返回如下JSON（没有则填"未知"或返回空数组）：
{
    "industryExperience": "行业经验描述，如'3年AI行业经验'，未知填未知",
    "managementExperience": "管理经验描述，如'带过5人团队'，未知填未知",
    "availableDate": "到岗时间，如'随时到岗'/'一周内'，未知填未知",
    "salaryExpectation": "薪资要求，如'25-30K'/'面议'，未知填未知",
    "portfolio": ["作品、GitHub、博客等链接或名称"],
    "certificates": ["职业证书、资格认证名称"]
}

只返回JSON。严格只提取原文明确出现的信息，不得推测。

## Prompt 3
你是资深HR分析师。请分析以下简历并给出综合评估，返回纯JSON。
[[position_hint]]

【简历文本】
[[text_12000]]

返回如下JSON：
{
    "overallScore": 0到100的整数,
    "keywords": ["关键词1", "关键词2", "关键词3", "关键词4", "关键词5"],
    "summary": "综合评价摘要（50-100字）",
    "positionMatch": "岗位匹配度分析",
    "experienceInsight": "经验洞察",
    "highlights": ["亮点1", "亮点2"],
    "risks": ["风险点1"],
    "recommendation": "推荐建议"
}

只返回JSON。注意：不要输出dimensions字段。keywords恰好5个，不足用技能补。

## Prompt 4
你是专业的简历解析器。请从简历文本中提取个人基本信息，返回纯JSON。
[[position_hint]]

【简历文本】
[[text_12000]]

返回如下JSON（找不到的填"未知"，应届经验填"应届"）：
{
    "name": "姓名",
    "gender": "男/女/未知",
    "age": null,
    "birthDate": "出生日期如2003-12-12，无则null",
    "education": "最高学历：大专/本科/硕士/博士/其他/未知",
    "experience": "工作年限如7年；无工作经历填在校中",
    "phone": "手机号，未知填未知",
    "email": "邮箱，未知填未知",
    "ethnicity": "民族，未提及填汉族",
    "nativePlace": "籍贯或现居地，未知填未知"
}

只返回JSON。严格要求：只能提取原文明确出现的信息，不得臆造。age留null，识别到出生日期填birthDate。

## Prompt 5
从简历文本中提取教育经历列表，返回纯JSON。
【简历文本】
[[text_12000]]

返回如下JSON（没有则返回空数组）：
{"educationHistory": [{"school": "学校", "degree": "学位", "major": "专业", "period": "时间段"}]}

只返回JSON。严格只提取原文明确出现的信息。

## Prompt 6
从简历文本中提取工作经历列表，返回纯JSON。
【简历文本】
[[text_12000]]

返回如下JSON（没有则返回空数组）：
{"workHistory": [{"company": "公司", "role": "职位", "period": "时间段", "description": "工作描述"}]}

只返回JSON。严格只提取原文明确出现的信息。

## Prompt 7
从简历文本中提取项目经历列表，返回纯JSON。
【简历文本】
[[text_12000]]

返回如下JSON（没有则返回空数组）：
{"projectHistory": [{"name": "项目名", "role": "角色", "period": "时间段", "description": "项目描述"}]}

只返回JSON。严格只提取原文明确出现的信息。

## Prompt 8
从简历文本中提取专业技能列表，返回纯JSON。
【简历文本】
[[text_12000]]

返回如下JSON（没有则返回空数组）：
{"skills": ["技能1", "技能2"]}

只返回JSON。严格只提取原文明确出现的技能。

## Prompt 9
你是资深HR分析师。请分析简历并给出评分和关键词，返回纯JSON。
[[position_hint]]

【简历文本】
[[text_12000]]

返回如下JSON：
{
    "overallScore": 0到100的整数,
    "keywords": ["关键词1", "关键词2", "关键词3", "关键词4", "关键词5"]
}

keywords恰好5个，不足用技能补。只返回JSON。不要输出其他字段。

## Prompt 10
你是资深HR分析师。请分析简历并给出洞察，返回纯JSON。
[[position_hint]]

【简历文本】
[[text_12000]]

返回如下JSON：
{
    "summary": "综合评价摘要（50-100字）",
    "positionMatch": "岗位匹配度分析",
    "experienceInsight": "经验洞察",
    "highlights": ["亮点1", "亮点2"],
    "risks": ["风险点1"],
    "recommendation": "推荐建议"
}

只返回JSON。不要输出overallScore或keywords字段。
