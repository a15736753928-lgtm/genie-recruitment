"""
Seed database with the 6 positions from 招聘与试用期管理手册.
Each position includes: JD, screening scorecard, interview scorecards (R1/R2),
week1 requirements, weeks 2-4 plan, later week scoring, and conversion criteria.
"""
import uuid
from datetime import date
from sqlalchemy import select
from app.database import async_session_factory
from app.models.recruitment import Position, PositionQuestion

# ══════════════════════════════════════════════════════════════
# Common scoring criteria (used by all positions)
# ══════════════════════════════════════════════════════════════

SCREENING_CRITERIA_COMMON = {
    "description": "简历筛选双维度评分表 (满分100分)",
    "dimensions": [
        {
            "group": "岗位匹配度",
            "weight": 0.60,
            "items": [
                {"name": "核心技术栈契合度", "weight": 0.30, "maxScore": 30,
                 "criteria": "完全匹配且有深入使用经验(27-30); 掌握大部分核心技术(21-26); 核心技术严重不匹配(0-20)"},
                {"name": "业务场景与项目经验", "weight": 0.20, "maxScore": 20,
                 "criteria": "主导过高度相关的复杂项目(18-20); 参与过类似项目(14-17); 无相关项目经验(0-13)"},
                {"name": "学历专业与优先条件", "weight": 0.10, "maxScore": 10,
                 "criteria": "对口专业且具备多项优先条件(9-10); 相关专业(7-8); 专业不符且无加分项(0-6)"},
            ]
        },
        {
            "group": "简历内容质量",
            "weight": 0.40,
            "items": [
                {"name": "项目描述与量化成果", "weight": 0.20, "maxScore": 20,
                 "criteria": "描述详实且成果量化明确(18-20); 描述完整但缺少量化数据(14-17); 描述空泛、堆砌名词(0-13)"},
                {"name": "履历真实性与连贯性", "weight": 0.10, "maxScore": 10,
                 "criteria": "履历稳定、成长清晰(9-10); 有可解释的短暂空窗(7-8); 频繁跳槽或存在明显疑点(0-6)"},
                {"name": "规范性与信息完整度", "weight": 0.10, "maxScore": 10,
                 "criteria": "排版专业、信息详实(9-10); 信息基本完整(7-8); 信息缺失严重、排版混乱(0-6)"},
            ]
        },
    ],
    "process": "AI初筛每日汇总→自动排序生成筛选报告→人工复核调整→取前N名安排面试(N由HR与用人部门共同确定,默认5)",
    "thresholds": {"talent_pool": 70, "daily_interview_max": 5}
}

INTERVIEW_CRITERIA_R1_COMMON = {
    "description": "第一轮面试评分表 (满分100分, ≥80分进入第二轮, 70-79分进入人才储备池)",
    "passThreshold": 80,
    "reserveThreshold": 70,
    "dimensions": [
        {"index": 1, "name": "岗位基础专业知识", "subtitle": "核心概念与原理广度", "maxScore": 10,
         "rubric": "准确阐述核心概念的基本定义与机制(3); 能指出该技术的适用场景或解决的痛点(3); 回答条理清晰,无常识性技术错误(2); 能主动延伸对比相关技术或底层原理(2)"},
        {"index": 2, "name": "岗位常规操作与排查", "subtitle": "常见场景实践基础", "maxScore": 10,
         "rubric": "排查思路清晰,步骤符合常规逻辑(3); 准确说出所需的具体命令、工具或方法(3); 考虑到边界条件(权限、网络、资源占用等)(2); 结合实际经验举例,而非背诵理论(2)"},
        {"index": 3, "name": "简历项目真实性", "subtitle": "核心项目深挖与细节验证", "maxScore": 10,
         "rubric": "能画出或清晰口述项目整体架构(3); 准确描述各组件的职责边界(2); 明确界定本人负责的模块(2); 说出关键数据指标(并发量、数据量等)且前后一致(3)"},
        {"index": 4, "name": "技术选型合理性", "subtitle": "选型原因与方案对比阐述", "maxScore": 10,
         "rubric": "说明选型时的业务背景(2); 说明核心约束条件(如成本、性能、团队能力)(2); 列举至少一种主流替代方案(2); 客观对比优劣势(2); 说出决定性的选型依据(2)"},
        {"index": 5, "name": "项目难点与突破", "subtitle": "难点攻克过程与量化成果", "maxScore": 10,
         "rubric": "难题描述具体,有技术深度(3); 排查过程有清晰的方法论:日志→定位→假设→验证(4); 最终解决手段切实可行并有量化收益(3)"},
        {"index": 6, "name": "语言沟通与表达", "subtitle": "表达清晰度与技术讲解能力", "maxScore": 10,
         "rubric": "三个词概括精准(3); 与其简历经历高度吻合(2); 每个词的一句话解释有具体事实支撑(3); 语言简洁流畅,表达自信自然(2)"},
        {"index": 7, "name": "逻辑思维与条理", "subtitle": "回答条理性与框架化思维", "maxScore": 10,
         "rubric": "步骤一:看监控告警确定影响面/止血(3); 步骤二:看日志/链路追踪定位具体报错节点(3); 步骤三:查核心指标/变更记录找根因(2); 回答结构化,条理分明(2)"},
        {"index": 8, "name": "临场反应与抗压", "subtitle": "追问质疑下的冷静应对", "maxScore": 10,
         "rubric": "面对质疑情绪稳定,不抵触(3); 能补充更深入的技术细节佐证(4); 若确有不足能坦诚承认并说明学习计划(3)"},
        {"index": 9, "name": "团队协作与角色", "subtitle": "跨职能协作与分歧解决", "maxScore": 10,
         "rubric": "角色定位清晰(如破冰者、执行者)(2); 有具体团队场景实例支撑(3); 说明该角色对团队产出的实际贡献(3); 展现与不同角色的协作经验(2)"},
        {"index": 10, "name": "求职意向与稳定性", "subtitle": "离职原因与职业规划契合", "maxScore": 10,
         "rubric": "离职原因客观合理(发展空间等)(3); 无负面抱怨(2); 表达对本岗位方向的明确兴趣点(3); 职业规划与本岗位成长路径吻合(2)"},
    ]
}

WEEK1_COMMON = {
    "description": "试用期第一周项目复现考核 (满分100分, ≥70分通过, <70分终止试用期)",
    "passThreshold": 70,
    "dimensions": [
        {"name": "项目复现完整度", "maxScore": 30,
         "criteria": "搭建公司标准开发环境并成功部署复现指派的历史项目; 核心链路跑通,功能与原项目一致"},
        {"name": "代码/方案还原度", "maxScore": 25,
         "criteria": "按原项目文档完成复现,输出与原项目一致的成果指标"},
        {"name": "独立解决问题能力", "maxScore": 25,
         "criteria": "复现过程中遇到环境报错、依赖冲突、配置异常等问题时,能独立排查解决,必要时与带教负责人有效沟通"},
        {"name": "规范性与总结", "maxScore": 20,
         "criteria": "复现过程符合公司开发规范,按时产出清晰完整的复现总结文档"},
    ]
}

LATER_WEEK_SCORING_COMMON = {
    "description": "后续周统一考核标准 (试用期结束时评定,满分100分)",
    "dimensions": [
        {"name": "交付质量", "weight": 0.40, "maxScore": 40,
         "criteria": "代码Bug率低,API接口稳定,部署无明显漏洞 (优秀35-40 / 合格25-34 / 较差<25)"},
        {"name": "交付效率", "weight": 0.30, "maxScore": 30,
         "criteria": "能在规定的排期内完成模块开发或部署任务,无无故延期 (优秀25-30 / 合格20-24 / 较差<20)"},
        {"name": "技术能力", "weight": 0.15, "maxScore": 15,
         "criteria": "对技术栈的应用熟练度,解决技术卡点的独立性 (优秀13-15 / 合格10-12 / 较差<10)"},
        {"name": "业务与沟通", "weight": 0.15, "maxScore": 15,
         "criteria": "准确理解业务痛点,遇到阻塞能主动沟通汇报,跨部门协作顺畅 (优秀13-15 / 合格10-12 / 较差<10)"},
    ]
}

CONVERSION_CRITERIA_COMMON = {
    "description": "试用期转正考核汇总表 (满分100分)",
    "thresholds": {"converted": 80, "extended": 70},
    "dimensions": [
        {"name": "后三周真实项目表现", "weight": 0.60,
         "description": "依据后三周任务的实际工作量与验收记录,由带教负责人在试用期结束时统一评定(满分100分)×60%",
         "assessor": "带教负责人"},
        {"name": "技术能力与业务产出", "weight": 0.20,
         "description": "试用期内提交的核心代码、文档、技术方案及解决的复杂问题数量与质量",
         "assessor": "带教负责人"},
        {"name": "团队协作与综合素养", "weight": 0.20,
         "description": "与研发/测试等上下游团队的沟通顺畅度、抗压能力、责任心及融入团队的程度",
         "assessor": "部门负责人"},
    ],
    "decisions": [
        {"range": "≥80分", "decision": "同意按期转正,进入公司人才库"},
        {"range": "70-79分", "decision": "延长试用期,需部门负责人审批"},
        {"range": "<70分", "decision": "不符合录用条件,予以辞退"},
    ]
}

# ══════════════════════════════════════════════════════════════
# Position-specific data
# ══════════════════════════════════════════════════════════════

POSITIONS_DATA = [
    {
        "name": "前沿部署工程师",
        "chapter_number": 5,
        "department": "AI基础设施部",
        "jd_responsibilities": (
            "1. 深入制造业生产、质量、供应链、仓储等一线场景,挖掘业务痛点,定位AI Agent落地场景;\n"
            "2. 联动客户业务、IT及产线团队,明确需求并输出可落地解决方案;\n"
            "3. 依托大模型与Agent平台,完成端到端方案设计、数据接入、系统集成、接口联调及上线迭代;\n"
            "4. 协同产品、算法及研发团队,沉淀行业模板、交付方法与通用产品能力;\n"
            "5. 推进项目落地交付,保障方案稳定可用、产生实际业务价值。"
        ),
        "jd_requirements": (
            "1. 软件工程、互联网工程相关专业,具备扎实工程开发与问题排查能力;\n"
            "2. 熟练掌握Python/Java/Go等至少一种主流编程语言;\n"
            "3. 熟悉API、数据库、数据处理、工作流编排及系统集成;\n"
            "4. 具备大模型、AI Agent、RAG等相关实操经验,可将AI能力转化为业务应用;\n"
            "5. 优秀沟通协调能力,可高效对接客户及内部团队,独立推进需求落地;\n"
            "6. 具备端到端项目Owner意识,适应需求变化,可接受出差/驻场。"
        ),
        "jd_preferred": (
            "1. 有制造业、MES/ERP/WMS/QMS等工业软件实操及数据处理经验;\n"
            "2. 有企业级AI项目落地、PoC验证及规模化复制经验;\n"
            "3. 具备复杂项目交付、多方协同及需求研判优化能力;\n"
            "4. 具备产品思维,可从项目中抽象通用化能力。"
        ),
        "jd_tech_stack": "Python/Java/Go、大模型(LLM)、AI Agent、RAG、API集成、工作流编排、数据库、MES/ERP/WMS/QMS",
        "week1_project_requirement": {
            "description": "复现公司已完成的某模型部署项目:按原项目文档在测试环境完成模型推理服务(如vLLM/TensorRT-LLM)的部署、量化与压测,输出与原项目一致的性能指标",
            "tasks": [
                "部署公司已完成的模型部署项目,在测试环境中复现模型推理服务(vLLM/TensorRT-LLM)的部署、量化与压测",
                "输出与原项目一致的性能指标(延迟/吞吐/显存占用等)",
                "撰写复现总结文档,说明部署路径、遇到的问题及解决过程"
            ]
        },
        "weeks_2_4_plan": {
            "weeks": [
                {"week": 2, "phase": "基础介入",
                 "tasks": ["熟悉现有Agent或大模型应用的部署流程", "独立完成一个非核心API接口的开发与测试", "参与一次线上问题的日志排查"]},
                {"week": 3, "phase": "模块开发",
                 "tasks": ["独立负责一个子模块的开发(如新增一个RAG检索策略或接入一个新工具)", "完成对应模块的Docker镜像构建与本地联调"]},
                {"week": 4, "phase": "核心协同",
                 "tasks": ["参与核心系统的集成或调优(如提升高并发下的响应速度)", "完成自己负责模块的代码Review,并输出一份部署维护SOP文档"]},
            ]
        },
        "interview_criteria_r2": {
            "description": "第二轮面试评分表 (满分100分, ≥85分予以录用)",
            "passThreshold": 85,
            "dimensions": [
                {"index": 1, "name": "大模型推理框架与优化", "subtitle": "vLLM/TensorRT-LLM与显存优化", "maxScore": 10,
                 "rubric": "解释vLLM的PagedAttention显存分页管理机制(3); 解释TGI的Continuous Batching机制(3); 解释TensorRT-LLM的图优化/量化内核(2); 对比三者在吞吐量与延迟上的权衡(2)"},
                {"index": 2, "name": "GPU算力调度与虚拟化", "subtitle": "MIG/MPS与算力隔离抢占", "maxScore": 10,
                 "rubric": "解释MIG在硬件层切分GPU的原理(4); 解释MPS的进程级共享机制(3); 对比两者在故障隔离和资源利用率上的差异(3)"},
                {"index": 3, "name": "模型量化与压缩技术", "subtitle": "AWQ/GPTQ与KV Cache量化", "maxScore": 10,
                 "rubric": "说明GPTQ基于二阶信息的量化思路(3); 说明AWQ按激活分布保护权重的思路(3); 说明SmoothQuant平滑变换的思路(2); 给出精度损失与加速比的综合对比(2)"},
                {"index": 4, "name": "分布式推理架构", "subtitle": "并行切分、扩缩容与高可用", "maxScore": 10,
                 "rubric": "负载均衡与请求队列设计(3); 基于QPS或队列长度的HPA扩缩容指标设计(3); 预热池与镜像预热等冷启动优化手段(2); 流式生成的连接保持机制(2)"},
                {"index": 5, "name": "K8s与容器化部署", "subtitle": "AI负载调度与镜像分发优化", "maxScore": 10,
                 "rubric": "镜像与模型权重分离(PVC/对象存储)方案(4); P2P镜像分发(Dragonfly等)加速原理(3); 镜像分层优化或lazy-loading(3)"},
                {"index": 6, "name": "高性能网络与通信", "subtitle": "NCCL/RDMA与集群网络调优", "maxScore": 10,
                 "rubric": "解释NCCL的Ring或Tree All-Reduce算法(3); 说明对NVLink/PCIe拓扑的自动探测(3); 解释GPUDirect RDMA零拷贝通信原理(4)"},
                {"index": 7, "name": "推理服务监控与压测", "subtitle": "关键指标、压测与链路追踪", "maxScore": 10,
                 "rubric": "TTFT(首Token延迟)指标监控(2); TPOT(每Token耗时)指标监控(2); KV Cache命中率与显存占用监控(3); 压测时区分输入输出长度分布(3)"},
                {"index": 8, "name": "边缘计算与端侧部署", "subtitle": "端侧优化与云边协同架构", "maxScore": 10,
                 "rubric": "模型量化(INT8/INT4)(3); 剪枝或蒸馏减小模型体积(2); 转换为端侧推理引擎(TensorRT/ONNX Runtime)(3); 算子融合优化(2)"},
                {"index": 9, "name": "算子优化与底层开发", "subtitle": "CUDA/Triton自定义算子", "maxScore": 10,
                 "rubric": "grid与block的划分(2); 内存层次设计(共享内存/寄存器)(3); 算子融合减少显存读写开销的原理(3); 规避bank conflict(2)"},
                {"index": 10, "name": "前沿AI基础设施视野", "subtitle": "行业趋势判断与技术洞察", "maxScore": 10,
                 "rubric": "说明硅基芯片(如Groq)的架构特点(3); 分析NVIDIA CUDA的生态壁垒(3); 给出对未来推理市场格局的判断(2); 结合推理成本趋势分析(2)"},
            ]
        },
    },
    {
        "name": "数据开发工程师",
        "chapter_number": 6,
        "department": "数据平台部",
        "jd_responsibilities": (
            "1. 负责公司业务系统中的数据开发工作,包括数据采集、清洗、加工、建模、同步与指标建设;\n"
            "2. 参与企业级数据中台、实时数据处理、数仓建设等相关工作;\n"
            "3. 基于Spark、Flink、Kafka等技术,完成离线数据处理和实时数据流处理任务;\n"
            "4. 参与ERP、生产、采购、库存、供应商等业务系统的数据流转设计;\n"
            "5. 配合后端系统开发,完成接口对接、数据落库、任务调度、数据校验、异常处理;\n"
            "6. 参与数据库表结构设计、SQL优化、Redis缓存设计及数据一致性处理;\n"
            "7. 与产品、前后端及业务人员协作,将业务需求转化为可落地的数据方案。"
        ),
        "jd_requirements": (
            "1. 计算机、软件工程、数据科学相关专业优先,具备扎实的数据开发或后端开发经验;\n"
            "2. 熟悉Spark,能独立完成离线数据处理、清洗、计算和任务优化;\n"
            "3. 使用过Flink或Kafka进行实时数据处理,理解消息队列、消费位点、异常重试等机制;\n"
            "4. 熟悉数仓建设,理解ODS/DWD/DWS/ADS分层思想;\n"
            "5. Java后端基础扎实,熟悉Spring Boot、MyBatis/MyBatis-Plus;\n"
            "6. 熟悉订单、采购、生产、库存等业务状态流转设计;\n"
            "7. 熟悉MySQL/PostgreSQL,具备SQL编写、索引设计和性能优化能力;\n"
            "8. 熟悉Redis缓存、分布式锁、热点数据处理等常见场景。"
        ),
        "jd_preferred": (
            "1. 有制造业、供应链、ERP/MES/WMS、采购、生产计划系统相关项目经验;\n"
            "2. 有实时数仓、数据中台、指标平台、经营分析系统建设经验;\n"
            "3. 既能做数据开发,又能参与Java后端业务开发;\n"
            "4. 对AI应用、企业智能化、数据驱动业务决策有兴趣。"
        ),
        "jd_tech_stack": "Spark、Flink、Kafka、MySQL/PostgreSQL、Redis、Java、Spring Boot、MyBatis、数仓分层(ODS/DWD/DWS/ADS)",
        "week1_project_requirement": {
            "description": "复现公司已完成的某数据管道项目:按原项目文档重建ETL链路(采集→清洗→入仓),跑通调度任务并产出与原项目一致的数据报表",
            "tasks": [
                "复现公司已完成的数据管道项目,重建ETL链路(采集→清洗→入仓)",
                "跑通调度任务,产出与原项目一致的数据报表",
                "撰写复现总结文档,说明管道设计、遇到的问题及解决过程"
            ]
        },
        "weeks_2_4_plan": {
            "weeks": [
                {"week": 2, "phase": "基础介入",
                 "tasks": ["接手现有的简单数据清洗脚本(Spark/Flink)", "独立完成1-2个简单维度的离线统计报表数据处理", "熟悉任务调度系统配置"]},
                {"week": 3, "phase": "模块开发",
                 "tasks": ["独立负责一条ODS到DWD的中等复杂度数据流开发", "处理数据倾斜或小文件问题", "编写数据校验(DQC)规则"]},
                {"week": 4, "phase": "核心协同",
                 "tasks": ["参与核心业务指标的数仓建模(如DWS层宽表设计)", "完成复杂SQL的性能调优", "梳理并输出数据流转血缘文档"]},
            ]
        },
        "interview_criteria_r2": {
            "description": "第二轮面试评分表 (满分100分, ≥85分予以录用)",
            "passThreshold": 85,
            "dimensions": [
                {"index": 1, "name": "离线数据处理与数仓建模", "subtitle": "维度建模与分层架构设计", "maxScore": 10,
                 "rubric": "星型模型:冗余换性能,查询简单(3); 雪花模型:规范化省存储,JOIN多性能差(3); 实际选择:分析多用星型/宽表(2); 结合ODS/DWD等分层说明(2)"},
                {"index": 2, "name": "实时计算与流处理架构", "subtitle": "Lambda/Kappa与Exactly-Once", "maxScore": 10,
                 "rubric": "Lambda:批流双链路,一致性高但维护成本高(4); Kappa:纯流式重放,代码一套但依赖长存储(4); 运维成本的综合对比结论(2)"},
                {"index": 3, "name": "Hadoop生态与Spark底层", "subtitle": "RDD机制、内存管理与Shuffle", "maxScore": 10,
                 "rubric": "窄依赖定义(一对一)(2); 宽依赖定义(引发Shuffle)(2); DAGScheduler按宽依赖切分Stage(3); 宽依赖对Task调度的影响及网络代价(3)"},
                {"index": 4, "name": "Flink核心机制与调优", "subtitle": "Checkpoint、状态与背压处理", "maxScore": 10,
                 "rubric": "Checkpoint触发与Barrier注入流程(3); Barrier对齐机制(3); Chandy-Lamport全局一致性快照原理(4)"},
                {"index": 5, "name": "数据湖与湖仓一体", "subtitle": "Iceberg/Hudi与Upsert优化", "maxScore": 10,
                 "rubric": "事务隔离实现机制对比(3); 并发更新(如Hudi COW/MOR)处理差异(3); Schema演进与分区演进支持度(2); 选型结论(2)"},
                {"index": 6, "name": "SQL优化与执行计划", "subtitle": "CBO、执行计划与等价改写", "maxScore": 10,
                 "rubric": "RBO:基于固定规则优化(2); CBO:基于统计信息估算代价(3); CBO利用行数/基数统计做Join重排序的过程(3); 统计信息过期导致的风险(2)"},
                {"index": 7, "name": "数据治理与数据质量", "subtitle": "血缘、质量监控与元数据", "maxScore": 10,
                 "rubric": "血缘系统整体架构(3); 难点1:动态SQL与临时表解析(3); 难点2:UDF与跨引擎语法解析(2); 字段级血缘的实现原理(2)"},
                {"index": 8, "name": "调度系统与数据集成", "subtitle": "DAG调度与实时增量同步", "maxScore": 10,
                 "rubric": "DAG定义方式对比(代码 vs 可视化)(3); 分布式调度架构对比(Scheduler单点 vs 去中心化)(4); 失败重试与补数能力对比(3)"},
                {"index": 9, "name": "大规模数据存储底层", "subtitle": "列存格式、LSM-Tree与分片", "maxScore": 10,
                 "rubric": "Parquet与ORC的存储结构(Row Group vs Stripe)(3); 编码方式差异(2); 压缩算法对比(2); 谓词下推利用统计信息跳过数据块的原理(3)"},
                {"index": 10, "name": "业务抽象与数据服务", "subtitle": "数据API、指标体系与价值量化", "maxScore": 10,
                 "rubric": "分层架构设计(查询引擎+预计算表)(3); 预计算+KV存储(Redis/HBase)承接高并发点查(4); 接口限流与降级策略(3)"},
            ]
        },
    },
    {
        "name": "知识图谱工程师",
        "chapter_number": 7,
        "department": "AI平台部",
        "jd_responsibilities": (
            "1. 服务架构:负责知识图谱服务端的架构设计与开发,利用FastAPI启动高性能服务,为后端Agent及下游业务提供标准化接口;\n"
            "2. 图数据检索:编写复杂SPARQL查询,实现图数据库(GraphDB/Neo4j)的高效检索与推理;\n"
            "3. 数据同步与集成:设计全量及增量同步方案,负责OBDA映射文件撰写,实现关系型数据库到RDF的逻辑映射;\n"
            "4. 本体治理与映射:配合本体专家在Protégé中维护本体模型,辅助撰写属性链及推理规则;\n"
            "5. 前瞻探索:参与非结构化数据挖掘,探索非结构化信息转化为知识图谱内容。"
        ),
        "jd_requirements": (
            "1. 精通Python,有FastAPI或类似框架(Flask/Django)实战项目经验;\n"
            "2. 精通SQL,熟练使用PostgreSQL或MySQL,了解Redis缓存机制;\n"
            "3. 熟悉图数据库原理,有GraphDB、Neo4j或gStore等至少一种图数据库使用经验;\n"
            "4. 掌握SPARQL或Cypher查询语言;\n"
            "5. 了解RDF、OWL、本体论等语义网基本概念。"
        ),
        "jd_preferred": (
            "1. 有知识图谱与Agent/大模型结合的应用经验;\n"
            "2. 有Protégé本体建模及OBDA实践经验;\n"
            "3. 有非结构化数据挖掘、NLP相关经验;\n"
            "4. 有高并发服务端架构设计经验。"
        ),
        "jd_tech_stack": "Python、FastAPI、SQL(PostgreSQL/MySQL)、Redis、GraphDB/Neo4j/gStore、SPARQL/Cypher、RDF/OWL、OBDA、Protégé",
        "week1_project_requirement": {
            "description": "复现公司已完成的知识图谱构建项目:实现实体抽取、关系构建和图谱查询(基于Neo4j),输出与原项目一致的图谱查询结果",
            "tasks": [
                "复现公司已完成的知识图谱构建项目,实现实体抽取、关系构建和图谱查询(基于Neo4j)",
                "输出与原项目一致的图谱查询结果",
                "撰写复现总结文档,说明构建流程、遇到的问题及解决过程"
            ]
        },
        "weeks_2_4_plan": {
            "weeks": [
                {"week": 2, "phase": "基础介入",
                 "tasks": ["熟悉公司图数据库的Schema设计", "完成几个简单的SPARQL/Cypher查询任务", "参与一次图谱数据的质量核查"]},
                {"week": 3, "phase": "模块开发",
                 "tasks": ["独立设计并实现一个小型本体模型的扩展", "完成对应数据的ETL入库和图谱构建"]},
                {"week": 4, "phase": "核心协同",
                 "tasks": ["参与图谱推理规则的优化", "完成图谱API接口的性能调优", "输出图谱数据维护SOP文档"]},
            ]
        },
        "interview_criteria_r2": {
            "description": "第二轮面试评分表 (满分100分, ≥85分予以录用)",
            "passThreshold": 85,
            "dimensions": [
                {"index": 1, "name": "图数据库底层与选型", "subtitle": "存储引擎架构与超级节点处理", "maxScore": 10,
                 "rubric": "对比Neo4j/NebulaGraph/HugeGraph存储架构差异(3); 解释Neo4j免索引邻接原理(3); 选型关键指标说明(2); 超级节点处理方案(2)"},
                {"index": 2, "name": "知识抽取与NLP", "subtitle": "NER/RE模型与小样本抽取", "maxScore": 10,
                 "rubric": "BiLSTM-CRF与BERT模型差异(3); 重叠关系/开放关系处理(2); 远程监督/小样本知识抽取策略(3); 联合抽取vs Pipeline(2)"},
                {"index": 3, "name": "知识融合与实体对齐", "subtitle": "消歧、对齐算法与海量数据匹配", "maxScore": 10,
                 "rubric": "实体消歧vs实体对齐算法流程(3); GNN在相似度计算中的作用(3); 冲突知识检测方案(2); 分块/LSH加速匹配原理(2)"},
                {"index": 4, "name": "图表示学习", "subtitle": "TransE系列、GNN与逻辑推理", "maxScore": 10,
                 "rubric": "TransE/TransH/TransR在多关系建模上的差异(3); GCN与GAT消息传递机制对比(3); 逻辑推理与表示学习的互补(2); 动态知识图谱时序建模(2)"},
                {"index": 5, "name": "图查询与性能优化", "subtitle": "Cypher/nGQL与深度遍历优化", "maxScore": 10,
                 "rubric": "复杂Cypher/Gremlin多跳查询编写(4); PROFILE查看执行计划优化(2); nGQL分布式执行模型(2); BFS深度遍历I/O优化(2)"},
                {"index": 6, "name": "大模型与知识图谱融合", "subtitle": "Graph RAG与幻觉检测", "maxScore": 10,
                 "rubric": "Graph RAG架构与传统RAG对比(3); LLM自动生成本体/Schema方案(3); 知识图谱用作LLM校验层实践(2); Text-to-Cypher微调技巧(2)"},
                {"index": 7, "name": "本体建模与治理", "subtitle": "本体设计、OWL与Schema演进", "maxScore": 10,
                 "rubric": "自顶向下本体设计方法论(3); OWL/RDFS类/属性/继承建模(2); Schema可扩展性设计(3); 粗细粒度平衡策略(2)"},
                {"index": 8, "name": "图计算与分布式", "subtitle": "Pregel模型与图算法优化", "maxScore": 10,
                 "rubric": "Spark GraphX vs Giraph BSP模型差异(3); PageRank/Louvain图算法实现(3); 图分区负载均衡(2); 外存图计算方案(2)"},
                {"index": 9, "name": "知识问答系统架构", "subtitle": "KBQA与多跳推理", "maxScore": 10,
                 "rubric": "KBQA全链路架构(3); NL问句到图谱路径映射(3); 多跳问答中间结果管理(2); 准确率与召回率优化方案(2)"},
                {"index": 10, "name": "图应用与行业落地", "subtitle": "成功案例、业务价值与ROI", "maxScore": 10,
                 "rubric": "风控/推荐等成功案例(3); 图项目的业务价值量化(2); Milestone与渐进式交付规划(3); 数据安全与隐私合规设计(2)"},
            ]
        },
    },
    {
        "name": "Agent工程师",
        "chapter_number": 8,
        "department": "AI平台部",
        "jd_responsibilities": (
            "1. 负责AI Agent应用架构设计与开发,包括Agent框架搭建、工具调用、RAG链路建设;\n"
            "2. 基于大语言模型(LLM)构建智能对话、任务规划与自动执行系统;\n"
            "3. 设计并优化Agent的Prompt工程、记忆管理和多Agent协作机制;\n"
            "4. 与产品、后端及业务团队协作,将业务需求转化为Agent可执行的智能流程;\n"
            "5. 跟踪前沿Agent技术,推动Agent能力的持续演进。"
        ),
        "jd_requirements": (
            "1. 计算机/AI相关专业,具备扎实的Python开发能力;\n"
            "2. 熟悉LangChain/LlamaIndex/AutoGPT等主流Agent框架;\n"
            "3. 深入理解RAG架构、向量数据库、Embedding等技术;\n"
            "4. 有LLM API调用经验(OpenAI/DeepSeek等),理解Prompt Engineering;\n"
            "5. 熟悉Function Calling/Tool Use机制;\n"
            "6. 具备良好的系统设计能力和工程化思维。"
        ),
        "jd_preferred": (
            "1. 有生产级Agent系统开发与运维经验;\n"
            "2. 熟悉多Agent协作框架(CrewAI/AutoGen等);\n"
            "3. 有前端开发能力,可独立完成Agent Demo展示;\n"
            "4. 有开源项目贡献或技术博客输出。"
        ),
        "jd_tech_stack": "Python、LangChain/LlamaIndex、RAG、向量数据库(Milvus/Qdrant)、LLM API、Function Calling、Prompt Engineering",
        "week1_project_requirement": {
            "description": "复现公司已完成的Agent应用项目:搭建Agent框架,连通工具调用和RAG链路,实现与原项目一致的对话能力和任务执行效果",
            "tasks": [
                "复现公司已完成的Agent应用项目,搭建Agent框架,连通工具调用和RAG链路",
                "实现与原项目一致的对话能力和任务执行效果,通过带教负责人验收",
                "撰写复现总结文档,说明架构设计、遇到的问题及解决过程"
            ]
        },
        "weeks_2_4_plan": {
            "weeks": [
                {"week": 2, "phase": "基础介入",
                 "tasks": ["熟悉公司Agent平台的架构和工具链", "独立完成一个Agent工具的接入(如新增API调用)", "参与一次Agent对话质量的评测"]},
                {"week": 3, "phase": "模块开发",
                 "tasks": ["独立负责一个Agent子模块的开发(如新增RAG检索策略)", "完成对应模块的测试和文档编写"]},
                {"week": 4, "phase": "核心协同",
                 "tasks": ["参与多Agent协作流程的设计", "完成Agent响应速度的优化", "输出Agent开发维护SOP文档"]},
            ]
        },
        "interview_criteria_r2": {
            "description": "第二轮面试评分表 (满分100分, ≥85分予以录用)",
            "passThreshold": 85,
            "dimensions": [
                {"index": 1, "name": "Agent架构与核心底层", "subtitle": "ReAct循环与工具调用栈", "maxScore": 10,
                 "rubric": "LangChain/AutoGPT/MetaGPT循环机制对比(3); LangGraph/LlamaIndex状态图执行原理(3); 错误处理与重试机制设计(2); 手写Agent核心循环能力(2)"},
                {"index": 2, "name": "LLM推理与调优", "subtitle": "模型选型、SFT与推理参数", "maxScore": 10,
                 "rubric": "不同参数量模型指令遵循能力对比(3); SFT微调专用模型方案(3); Temperature/Top-P对规划确定性的影响(2); DPO/RLHF偏好对齐(2)"},
                {"index": 3, "name": "工具调用与函数执行", "subtitle": "Function Calling与安全护栏", "maxScore": 10,
                 "rubric": "OpenAI Function Calling JSON Schema约束机制(3); 工具检索避免超Context Limit(3); 失败信息有效反馈给模型(2); Human-in-the-loop安全机制(2)"},
                {"index": 4, "name": "记忆与会话管理", "subtitle": "短期/长期记忆与上下文压缩", "maxScore": 10,
                 "rubric": "短期vs长期记忆实现方式(3); 向量数据库存储情景记忆(3); 上下文压缩/摘要策略(2); MemGPT分层存储架构(2)"},
                {"index": 5, "name": "多Agent协作与规划", "subtitle": "任务分解、通信与规划算法", "maxScore": 10,
                 "rubric": "Camel/AutoGen/CrewAI多Agent框架对比(3); 避免信息回音室策略(2); ToT/GoT规划算法应用(3); Manager Agent动态任务分配(2)"},
                {"index": 6, "name": "RAG与知识增强检索", "subtitle": "混合检索与重排序", "maxScore": 10,
                 "rubric": "Query Rewrite与Query Routing机制(3); BM25+Dense混合检索优化(3); Reranking提升准确率(2); 复杂文档解析与分块策略(2)"},
                {"index": 7, "name": "Prompt工程与系统指令", "subtitle": "System Prompt与注入攻击", "maxScore": 10,
                 "rubric": "结构化System Prompt设计(3); Few-Shot提升特定场景表现(2); 精确指令与探索空间平衡(2); Prompt注入攻防策略(3)"},
                {"index": 8, "name": "Agent评测与安全合规", "subtitle": "评测基准与自动化评测", "maxScore": 10,
                 "rubric": "AgentBench/WebArena等评测基准(3); 业务场景自动化评测管线(2); 实时安全/合规/事实性审核(3); 逻辑一致性与可解释性(2)"},
                {"index": 9, "name": "工程化与可观测性", "subtitle": "容器化、会话状态与可观测", "maxScore": 10,
                 "rubric": "Python Agent容器化到K8s(2); Redis维护高并发会话状态(3); 日志/指标/链路追踪三件套(3); 流式输出降低感知延迟(2)"},
                {"index": 10, "name": "前沿Agent研究与视野", "subtitle": "前沿动态与未来判断", "maxScore": 10,
                 "rubric": "RL+LLM Agent融合趋势(3); 顶会论文解读能力(2); Embodied AI与纯Agent差异(2); Agent落地工业的最大障碍分析(3)"},
            ]
        },
    },
    {
        "name": "全栈开发工程师",
        "chapter_number": 9,
        "department": "技术研发部",
        "jd_responsibilities": (
            "1. 负责公司业务系统的前后端全栈开发,包括Web前端页面、后端API、数据库设计;\n"
            "2. 参与系统架构设计、技术选型和代码评审;\n"
            "3. 完成前后端接口联调,保障系统高性能、高可用;\n"
            "4. 与产品、UI/UX及业务团队协作,将需求转化为高质量的技术方案;\n"
            "5. 持续优化代码质量、性能和用户体验。"
        ),
        "jd_requirements": (
            "1. 计算机/软件工程相关专业;\n"
            "2. 前端:熟练掌握React/Vue等主流框架,了解Virtual DOM原理;\n"
            "3. 后端:熟练掌握至少一种后端语言(Java/Python/Go/Node.js)及相关框架;\n"
            "4. 熟悉关系型数据库(MySQL/PostgreSQL)和Redis缓存;\n"
            "5. 了解Docker、CI/CD等DevOps实践;\n"
            "6. 具备良好的代码规范和文档习惯。"
        ),
        "jd_preferred": (
            "1. 有从0到1的全栈项目开发经验;\n"
            "2. 熟悉微服务架构和分布式系统设计;\n"
            "3. 有TypeScript、Next.js等现代化前端技术栈经验;\n"
            "4. 有开源项目或个人技术作品。"
        ),
        "jd_tech_stack": "React/Vue、TypeScript、Java/Python/Go/Node.js、Spring Boot/FastAPI、MySQL/PostgreSQL、Redis、Docker、Git",
        "week1_project_requirement": {
            "description": "复现公司已完成的Web应用项目:还原前后端完整功能,通过带教负责人验收",
            "tasks": [
                "复现公司已完成的Web应用项目,还原前后端完整功能",
                "实现与原项目一致的核心页面和接口功能,通过带教负责人验收",
                "撰写复现总结文档,说明全栈技术选择、遇到的问题及解决过程"
            ]
        },
        "weeks_2_4_plan": {
            "weeks": [
                {"week": 2, "phase": "基础介入",
                 "tasks": ["熟悉公司项目的技术栈和代码规范", "独立完成一个简单CRUD功能的前后端开发", "参与一次代码Review"]},
                {"week": 3, "phase": "模块开发",
                 "tasks": ["独立负责一个业务模块的全栈开发", "完成对应模块的单元测试和接口文档"]},
                {"week": 4, "phase": "核心协同",
                 "tasks": ["参与核心业务模块的性能优化", "完成前端关键渲染路径的优化", "输出模块的部署维护文档"]},
            ]
        },
        "interview_criteria_r2": {
            "description": "第二轮面试评分表 (满分100分, ≥85分予以录用)",
            "passThreshold": 85,
            "dimensions": [
                {"index": 1, "name": "前端框架底层与性能", "subtitle": "Fiber/响应式原理与渲染优化", "maxScore": 10,
                 "rubric": "React Fiber/Vue3响应式系统底层原理(3); 内存泄漏定位方案(3); 关键渲染路径与回流/重绘优化(2); SSR/SSG/CSR选型对比(2)"},
                {"index": 2, "name": "前端工程化与架构", "subtitle": "构建工具、微前端与规范", "maxScore": 10,
                 "rubric": "Webpack/Vite/Rollup构建机制对比(3); 微前端架构设计(3); 代码规范与自动化测试(2); 大型前端项目重构规划(2)"},
                {"index": 3, "name": "后端框架底层与并发", "subtitle": "I/O模型、并发编程与一致性", "maxScore": 10,
                 "rubric": "Spring Boot/Node.js/Go核心机制(3); AQS/ReentrantLock原理(2); 高并发本地缓存设计(3); Goroutine vs 线程池 vs Event Loop(2)"},
                {"index": 4, "name": "微服务架构与分布式", "subtitle": "分布式事务、服务治理与链路追踪", "maxScore": 10,
                 "rubric": "2PC/TCC/Saga分布式事务对比(3); Service Mesh与传统微服务对比(3); 高可用API网关设计(2); 分布式链路追踪定位(2)"},
                {"index": 5, "name": "数据库底层与SQL调优", "subtitle": "B+树、MVCC与分库分表", "maxScore": 10,
                 "rubric": "B+树结构与聚簇/非聚簇索引(3); MVCC多版本并发控制机制(2); 分库分表与跨分片查询(3); EXPLAIN执行计划分析(2)"},
                {"index": 6, "name": "缓存技术与Redis深度", "subtitle": "高并发穿透/击穿/雪崩与分布式锁", "maxScore": 10,
                 "rubric": "Redis单线程模型与多线程(3); 缓存穿透/击穿/雪崩解决方案(3); RDB/AOF持久化机制对比(2); Redlock分布式锁原理(2)"},
                {"index": 7, "name": "消息队列与异步解耦", "subtitle": "可靠性原理与顺序消息", "maxScore": 10,
                 "rubric": "Kafka/RabbitMQ/RocketMQ底层存储模型对比(3); 消息顺序消费方案(3); 消息丢失/重复幂等处理(2); 积压快速消费恢复(2)"},
                {"index": 8, "name": "网络协议与安全", "subtitle": "TCP/HTTPS与XSS/CSRF防御", "maxScore": 10,
                 "rubric": "TCP三次握手四次挥手TIME_WAIT(3); HTTPS加密握手过程(3); XSS/CSRF防御方案(2); OAuth 2.0/JWT鉴权体系(2)"},
                {"index": 9, "name": "DevOps与CI/CD", "subtitle": "Docker/K8s与发布流水线", "maxScore": 10,
                 "rubric": "Docker Namespace/Cgroups隔离原理(3); Pod创建到调度全流程(2); 高效Dockerfile编写(2); CI/CD全流水线设计(3)"},
                {"index": 10, "name": "系统架构与容量规划", "subtitle": "可扩展架构、ID生成与权限设计", "maxScore": 10,
                 "rubric": "千万用户可扩展架构设计(3); 秒杀系统架构与压力评估(3); 全局唯一ID方案对比(2); OpenAPI版本/安全/限流设计(2)"},
            ]
        },
    },
    {
        "name": "高级运维工程师",
        "chapter_number": 10,
        "department": "基础设施部",
        "jd_responsibilities": (
            "1. 负责K8s/Docker集群的日常运维、监控和故障处理;\n"
            "2. 设计和维护CI/CD流水线,推动自动化部署和发布流程;\n"
            "3. 建设监控告警体系(Prometheus/Grafana/ELK),保障系统可观测性;\n"
            "4. 负责Linux系统调优、网络故障排查和安全加固;\n"
            "5. 编写运维自动化和基础设施即代码(IaC)方案;\n"
            "6. 参与on-call轮值,快速响应和解决生产环境问题。"
        ),
        "jd_requirements": (
            "1. 计算机相关专业,具备扎实的Linux系统管理能力;\n"
            "2. 深入理解Kubernetes架构,有生产级K8s集群运维经验;\n"
            "3. 熟悉Docker容器技术,理解Namespace/Cgroups等底层机制;\n"
            "4. 掌握至少一种CI/CD工具(Jenkins/GitLab CI);\n"
            "5. 熟悉Prometheus、Grafana、ELK等监控日志系统;\n"
            "6. 具备Shell/Python脚本编写能力;\n"
            "7. 了解主流数据库(MySQL/PostgreSQL/Redis)的运维。"
        ),
        "jd_preferred": (
            "1. 有大规模(百台以上)K8s集群运维经验;\n"
            "2. 熟悉Terraform/Ansible等IaC工具;\n"
            "3. 有云平台(AWS/Azure/阿里云)运维经验;\n"
            "4. 有SRE转型或实践经历。"
        ),
        "jd_tech_stack": "Kubernetes、Docker、Linux、Prometheus/Grafana/ELK、Jenkins/GitLab CI、Shell/Python、Nginx、MySQL/PostgreSQL/Redis",
        "week1_project_requirement": {
            "description": "复现公司已完成的服务架构:搭建K8s集群、CI/CD流水线并配置监控告警链路,实现与原架构一致的运维能力和监控效果",
            "tasks": [
                "复现公司已完成的服务架构,搭建K8s集群、CI/CD流水线并配置监控告警链路",
                "实现与原架构一致的运维能力和监控效果,链路通过带教负责人验收",
                "撰写复现总结文档,说明架构组成、遇到的问题及解决过程"
            ]
        },
        "weeks_2_4_plan": {
            "weeks": [
                {"week": 2, "phase": "基础介入",
                 "tasks": ["熟悉公司现有基础设施和运维流程", "独立完成一个监控告警规则的添加", "参与一次线上故障的排查与复盘"]},
                {"week": 3, "phase": "模块开发",
                 "tasks": ["独立负责一条CI/CD流水线的优化", "完成日志采集与分析管道的搭建"]},
                {"week": 4, "phase": "核心协同",
                 "tasks": ["参与K8s集群的容量规划与扩容", "完成运维自动化的脚本编写", "输出基础设施运维SOP文档"]},
            ]
        },
        "interview_criteria_r2": {
            "description": "第二轮面试评分表 (满分100分, ≥85分予以录用)",
            "passThreshold": 85,
            "dimensions": [
                {"index": 1, "name": "K8s集群运维与架构", "subtitle": "高可用集群与CNI网络模型调优", "maxScore": 10,
                 "rubric": "高可用K8s集群架构(3); Pod Pending/CrashLoopBackOff排查(3); Calico/Flannel网络模型性能对比(2); Resource Quota/LimitRange配置(2)"},
                {"index": 2, "name": "容器化底层技术", "subtitle": "Docker Namespace/Cgroups与镜像优化", "maxScore": 10,
                 "rubric": "Namespace/Cgroups隔离原理(3); 容器磁盘空间清理(2); Docker Bridge iptables/veth机制(3); 多阶段构建与层缓存优化(2)"},
                {"index": 3, "name": "Linux系统与内核调优", "subtitle": "启动流程、性能排查与TCP调优", "maxScore": 10,
                 "rubric": "Linux启动流程与故障排查(3); Load Average高而CPU低的定位(3); OOM Killer机制与保护策略(2); sysctl.conf TCP参数调优(2)"},
                {"index": 4, "name": "高可用架构与负载均衡", "subtitle": "LVS/Nginx/Keepalived方案", "maxScore": 10,
                 "rubric": "Nginx/HAProxy/LVS四七层对比(3); Keepalived VRRP协议与脑裂处理(3); Nginx worker调优(2); 多AZ/多Region容灾架构(2)"},
                {"index": 5, "name": "CI/CD与自动化运维", "subtitle": "发布流水线与灰度策略", "maxScore": 10,
                 "rubric": "GitLab CI/Jenkins全流水线设计(3); 蓝绿/金丝雀滚动发布策略(3); 流水线加速优化(2); 凭证安全管理(2)"},
                {"index": 6, "name": "监控告警与日志体系", "subtitle": "Prometheus/ELK/Loki与降噪", "maxScore": 10,
                 "rubric": "Prometheus+Grafana+Alertmanager架构(3); PromQL编写P99延迟/错误率(2); ELK vs Loki日志方案对比(3); 告警降噪与升级策略(2)"},
                {"index": 7, "name": "数据库与中间件运维", "subtitle": "主从延迟、集群故障与DDL风险", "maxScore": 10,
                 "rubric": "MySQL主从延迟排查优化(3); Redis Cluster故障转移Gossip协议(3); Kafka Broker宕机Leader选举(2); pt-osc在线DDL(2)"},
                {"index": 8, "name": "云平台架构与成本优化", "subtitle": "VPC设计、弹性伸缩与成本控制", "maxScore": 10,
                 "rubric": "Well-Architected VPC网络设计(3); Spot Instances+CA降低成本(3); 全局资源成本扫描优化(2); 混合云专线延迟/高可用(2)"},
                {"index": 9, "name": "故障排查与应急响应", "subtitle": "P0故障处理、抓包分析与复盘", "maxScore": 10,
                 "rubric": "P0故障从发现到复盘全流程(3); 502 Bad Gateway前5分钟排查(3); tcpdump/Wireshark/mtr抓包分析(2); COE故障复盘报告编写(2)"},
                {"index": 10, "name": "IaC与运维工程化", "subtitle": "Terraform/Ansible与SRE转型", "maxScore": 10,
                 "rubric": "IaC核心理念与Terraform State管理(3); Ansible千台级执行优化(2); 运维脚本工程化实践(3); Packer+Terraform不可变基础设施(2)"},
            ]
        },
    },
]

# ══════════════════════════════════════════════════════════════
# Question banks per position (first round)
# ══════════════════════════════════════════════════════════════

QUESTION_BANKS = {
    "前沿部署工程师": {
        "first": [
            # Q1: 岗位基础专业知识
            {"index": 1, "category": "岗位基础专业知识", "difficulty": "medium",
             "content": "1.1 请简述 Docker 和 Docker Compose 的核心概念与常用场景。"},
            {"index": 2, "category": "岗位基础专业知识", "difficulty": "medium",
             "content": "1.2 什么是容器化和虚拟化？容器和虚拟机有哪些区别？"},
            {"index": 3, "category": "岗位基础专业知识", "difficulty": "medium",
             "content": "1.3 在 Linux 中，如何查看当前系统的端口占用情况和进程运行状态？"},
            {"index": 4, "category": "岗位基础专业知识", "difficulty": "medium",
             "content": "1.4 大语言模型（LLM）在推理时，显存占用主要由哪几个部分组成？"},
            {"index": 5, "category": "岗位基础专业知识", "difficulty": "hard",
             "content": "1.5 什么是分布式推理？多卡推理和多机推理的区别是什么？"},
            # Q2: 常规操作与排查
            {"index": 6, "category": "岗位常规操作与排查", "difficulty": "medium",
             "content": "2.1 客户现场提供了一台服务器，无外网连接，你需要将几十GB的模型镜像文件部署上去，流程是什么？"},
            {"index": 7, "category": "岗位常规操作与排查", "difficulty": "medium",
             "content": "2.2 部署一个 Python 推理服务时，发现缺少某个 .so 动态链接库，你会如何排查和解决？"},
            {"index": 8, "category": "岗位常规操作与排查", "difficulty": "medium",
             "content": "2.3 客户反馈推理系统延迟过高，不适合生产使用，你会从哪些指标入手排查？"},
            {"index": 9, "category": "岗位常规操作与排查", "difficulty": "medium",
             "content": "2.4 现场部署时，发现端口冲突，占用端口的进程是客户的核心业务，你如何处理？"},
            {"index": 10, "category": "岗位常规操作与排查", "difficulty": "hard",
             "content": "2.5 如何编写一个 Shell 脚本，实现服务启动、健康检查和自动重启？"},
        ],
    },
    "数据开发工程师": {
        "first": [
            {"index": 1, "category": "岗位基础专业知识", "difficulty": "medium",
             "content": "1.1 请简述数据仓库中 ODS、DWD、DWS、ADS 各层的基本逻辑和作用。"},
            {"index": 2, "category": "岗位基础专业知识", "difficulty": "medium",
             "content": "1.2 什么是数据倾斜？MapReduce 和 Spark 中通常是什么原因导致的？"},
            {"index": 3, "category": "岗位基础专业知识", "difficulty": "medium",
             "content": "1.3 简述 Hive 内部表（Managed Table）和外部表（External Table）的区别。"},
            {"index": 4, "category": "岗位基础专业知识", "difficulty": "medium",
             "content": "1.4 什么是维表？什么是事实表？星型模型和雪花模型有什么区别？"},
            {"index": 5, "category": "岗位基础专业知识", "difficulty": "hard",
             "content": "1.5 简述批处理和实时流处理的核���差异及典型技术栈。"},
            {"index": 6, "category": "岗位常规操作与排查", "difficulty": "medium",
             "content": "2.1 写一条 SQL，找出某张订单表中，每个用户最近一次购买的产品名称和时间。"},
            {"index": 7, "category": "岗位常规操作与排查", "difficulty": "medium",
             "content": "2.2 发现某个凌晨时段的 ETL 任务执行时间比平时长了 3 倍，你会如何排查？"},
            {"index": 8, "category": "岗位常规操作与排查", "difficulty": "medium",
             "content": "2.3 如果上游业务修改了字段类型，导致数据同步任务报错，你会如何处理？"},
            {"index": 9, "category": "岗位常规操作与排查", "difficulty": "medium",
             "content": "2.4 在处理包含 JSON 字符串字段的日志数据时，通常通过什么方法进行解析和提取？"},
            {"index": 10, "category": "岗位常规操作与排查", "difficulty": "hard",
             "content": "2.5 你如何验证你开发的数仓报表和数据源的数据准确性？"},
        ],
    },
    "知识图谱工程师": {
        "first": [
            {"index": 1, "category": "岗位基础专业知识", "difficulty": "medium",
             "content": "1.1 什么是知识图谱？请用通俗的语言解释实体、属性和关系。"},
            {"index": 2, "category": "岗位基础专业知识", "difficulty": "medium",
             "content": "1.2 简述图数据库（如Neo4j）和传统关系型数据库在存储和查询上的本质区别。"},
            {"index": 3, "category": "岗位基础专业知识", "difficulty": "medium",
             "content": "1.3 什么是命名实体识别（NER）和关系抽取（RE）？"},
            {"index": 4, "category": "岗位基础专业知识", "difficulty": "medium",
             "content": "1.4 在知识图谱中，什么是知识融合（实体对齐）？为什么需要这一步？"},
            {"index": 5, "category": "岗位基础专业知识", "difficulty": "hard",
             "content": "1.5 列举几种常用的知识图谱表示学习（Graph Embedding）方法。"},
            {"index": 6, "category": "岗位常规操作与排查", "difficulty": "medium",
             "content": "2.1 给一段非结构化文本，你需要通过什么流程将其转化为图数据库中的结构化数据？"},
            {"index": 7, "category": "岗位常规操作与排查", "difficulty": "medium",
             "content": "2.2 使用 Cypher 或 Gremlin 查询图数据库时，如何查找两个节点之间的最短路径？"},
            {"index": 8, "category": "岗位常规操作与排查", "difficulty": "medium",
             "content": "2.3 如果抽取的实体中存在大量同义词(如\"苹果公司\"和\"Apple Inc.\")，你如何处理？"},
            {"index": 9, "category": "岗位常规操作与排查", "difficulty": "medium",
             "content": "2.4 图数据库查询某个复杂关系时非常缓慢，你会如何优化？"},
            {"index": 10, "category": "岗位常规操作与排查", "difficulty": "hard",
             "content": "2.5 使用什么方法评估你构建的知识图谱的数据准确率？"},
        ],
    },
    "Agent工程师": {
        "first": [
            {"index": 1, "category": "岗位基础专业知识", "difficulty": "medium",
             "content": "1.1 什么是 AI Agent？与传统 LLM 对话系统有什么本质区别？"},
            {"index": 2, "category": "岗位基础专业知识", "difficulty": "medium",
             "content": "1.2 简述 ReAct（Reasoning and Acting）框架的工作原理。"},
            {"index": 3, "category": "岗位基础专业知识", "difficulty": "medium",
             "content": "1.3 什么是向量数据库？在 RAG架构中扮演什么角色？"},
            {"index": 4, "category": "岗位基础专业知识", "difficulty": "medium",
             "content": "1.4 调用大模型 API 时，Temperature 和 Top-p 参数分别控制什么？"},
            {"index": 5, "category": "岗位基础专业知识", "difficulty": "hard",
             "content": "1.5 什么是 Prompt Engineering？有哪些常用的技巧？"},
            {"index": 6, "category": "岗位常规操作与排查", "difficulty": "medium",
             "content": "2.1 Agent 执行任务时出现循环、无限制调用某个工具，你会如何排查和解决？"},
            {"index": 7, "category": "岗位常规操作与排查", "difficulty": "medium",
             "content": "2.2 RAG 系统检索到的文档片段与用户问题不相关，你会在哪些环节进行优化？"},
            {"index": 8, "category": "岗位常规操作与排查", "difficulty": "medium",
             "content": "2.3 Agent 调用外部 API 时遇到了超时或报错，你如何在代码中处理使其更健壮？"},
            {"index": 9, "category": "岗位常规操作与排查", "difficulty": "medium",
             "content": "2.4 如果模型返回的 JSON 格式不规范导致解析失败，你有什么解决办法？"},
            {"index": 10, "category": "岗位常规操作与排查", "difficulty": "hard",
             "content": "2.5 如果要你实现一个自动读取邮件附件并总结的AI Agent，你会如何规划其模块？"},
        ],
    },
    "全栈开发工程师": {
        "first": [
            {"index": 1, "category": "岗位基础专业知识", "difficulty": "medium",
             "content": "1.1 简述前端 Vue/React 中虚拟 DOM（Virtual DOM）的概念和作用。"},
            {"index": 2, "category": "岗位基础专业知识", "difficulty": "medium",
             "content": "1.2 什么是跨域（CORS）？前后分离架构下通常通过哪些方式解决跨域问题？"},
            {"index": 3, "category": "岗位基础专业知识", "difficulty": "medium",
             "content": "1.3 简述 HTTP 协议中 GET 和 POST 的主要区别。"},
            {"index": 4, "category": "岗位基础专业知识", "difficulty": "medium",
             "content": "1.4 什么是 JWT？它由哪几个部分组成？"},
            {"index": 5, "category": "岗位基础专业知识", "difficulty": "hard",
             "content": "1.5 在关系型数据库中，索引是什么？索引过多会有什么影响？"},
            {"index": 6, "category": "岗位常规操作与排查", "difficulty": "medium",
             "content": "2.1 用户反馈页面点击按钮没有响应，你会如何使用浏览器开发者工具进行排查？"},
            {"index": 7, "category": "岗位常规操作与排查", "difficulty": "medium",
             "content": "2.2 前端页面加载非常慢（白屏时间长），你会从哪些方面入手优化？"},
            {"index": 8, "category": "岗位常规操作与排查", "difficulty": "medium",
             "content": "2.3 后端接口返回 500 或 502 错误，作为全栈开发你如何定位问题？"},
            {"index": 9, "category": "岗位常规操作与排查", "difficulty": "medium",
             "content": "2.4 设计一个用户注册登录模块时，你如何保证密码的安全存储和传输？"},
            {"index": 10, "category": "岗位常规操作与排查", "difficulty": "hard",
             "content": "2.5 如果需要实现一个前端实时展示数据变化的仪表盘，你会选择什么技术方案（轮询/WebSocket/SSE）？"},
        ],
    },
    "高级运维工程师": {
        "first": [
            {"index": 1, "category": "岗位基础专业知识", "difficulty": "medium",
             "content": "1.1 简述 K8s 中 Pod 的生命周期及其关键状态。"},
            {"index": 2, "category": "岗位基础专业知识", "difficulty": "medium",
             "content": "1.2 Linux 系统中，软链接和硬链接有什么区别？"},
            {"index": 3, "category": "岗位基础专业知识", "difficulty": "medium",
             "content": "1.3 Docker 容器镜像如何分层？什么是写时复制（CoW）？"},
            {"index": 4, "category": "岗位基础专业知识", "difficulty": "medium",
             "content": "1.4 简述什么是 CI/CD？在运维过程中它解决了什么痛点？"},
            {"index": 5, "category": "岗位基础专业知识", "difficulty": "hard",
             "content": "1.5 常见的负载均衡算法有哪些？Nginx 默认使用的是哪种？"},
            {"index": 6, "category": "岗位常规操作与排查", "difficulty": "medium",
             "content": "2.1 一台 Linux 服务器突然响应变慢，你会按什么顺序排查？"},
            {"index": 7, "category": "岗位常规操作与排查", "difficulty": "medium",
             "content": "2.2 发现某个容器频繁重启（CrashLoopBackOff），你使用哪些命令查看原因？"},
            {"index": 8, "category": "岗位常规操作与排查", "difficulty": "medium",
             "content": "2.3 数据库（如 MySQL）CPU 突然飙高，通常是什么原因？如何快速定位？"},
            {"index": 9, "category": "岗位常规操作与排查", "difficulty": "medium",
             "content": "2.4 使用 Nginx 做反向代理时，如何获取客户端的真实 IP？"},
            {"index": 10, "category": "岗位常规操作与排查", "difficulty": "hard",
             "content": "2.5 如果你要定期备份某个目录下的文件并清理超过7天的文件，你会怎么写脚本并设置定时任务？"},
        ],
    },
}


async def seed_positions_v2(create_missing: bool = True):
    """Seed handbook positions.

    create_missing=False 时只更新仍存在的岗位，不会把用户已删除的岗位再插回库。
    """
    async with async_session_factory() as db:
        for pos_data in POSITIONS_DATA:
            result = await db.execute(
                select(Position).where(Position.name == pos_data["name"])
            )
            existing = result.scalar_one_or_none()

            if existing:
                # Update existing position
                for key, value in pos_data.items():
                    if key == "name":
                        continue
                    setattr(existing, key, value)
                # Set common criteria
                existing.screening_criteria = SCREENING_CRITERIA_COMMON
                existing.interview_criteria_r1 = INTERVIEW_CRITERIA_R1_COMMON
                existing.later_week_scoring = LATER_WEEK_SCORING_COMMON
                existing.conversion_criteria = CONVERSION_CRITERIA_COMMON
                position = existing
                print(f"Updated: {pos_data['name']}")
            elif create_missing:
                # Create new position (first install only)
                position = Position(
                    name=pos_data["name"],
                    **{k: v for k, v in pos_data.items() if k != "name"},
                    screening_criteria=SCREENING_CRITERIA_COMMON,
                    interview_criteria_r1=INTERVIEW_CRITERIA_R1_COMMON,
                    later_week_scoring=LATER_WEEK_SCORING_COMMON,
                    conversion_criteria=CONVERSION_CRITERIA_COMMON,
                )
                db.add(position)
                await db.flush()
                await db.refresh(position)
                print(f"Created: {pos_data['name']}")
            else:
                print(f"Skipped (not recreating deleted): {pos_data['name']}")
                continue

            # ── Seed question banks ──
            qb = QUESTION_BANKS.get(pos_data["name"], {})
            for round_key in ("first",):
                if round_key not in qb:
                    continue
                # Check if questions already exist
                existing_qs = await db.execute(
                    select(PositionQuestion).where(
                        PositionQuestion.position_id == position.id,
                        PositionQuestion.round == round_key,
                    )
                )
                if existing_qs.scalars().first():
                    continue  # Skip if already seeded

                for q_data in qb[round_key]:
                    db.add(PositionQuestion(
                        position_id=position.id,
                        round=round_key,
                        index_num=q_data["index"],
                        content=q_data["content"],
                        category=q_data["category"],
                        difficulty=q_data["difficulty"],
                    ))
                print(f"  Seeded {len(qb[round_key])} questions for {round_key} round")

        # ── Update old default positions with Common criteria ──
        old_positions = await db.execute(
            select(Position).where(Position.screening_criteria.is_(None))
        )
        for old_pos in old_positions.scalars().all():
            old_pos.screening_criteria = SCREENING_CRITERIA_COMMON
            old_pos.interview_criteria_r1 = INTERVIEW_CRITERIA_R1_COMMON
            old_pos.later_week_scoring = LATER_WEEK_SCORING_COMMON
            old_pos.conversion_criteria = CONVERSION_CRITERIA_COMMON
            print(f"Updated legacy position with criteria: {old_pos.name}")

        await db.commit()

    print("\n[OK] Position seeding complete.")


if __name__ == "__main__":
    import asyncio
    asyncio.run(seed_positions_v2())
