#!/bin/bash
# Genie Recruitment System - Startup Script

echo "=========================================="
echo "  Genie 智能招聘系统 - 后端服务"
echo "=========================================="

# Check PostgreSQL
echo "[1/4] 检查 PostgreSQL 连接..."
PGPASSWORD=postgres psql -h localhost -p 5432 -U postgres -d postgres -c "SELECT 1;" > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "  ✓ PostgreSQL 连接正常"
else
    echo "  ✗ PostgreSQL 连接失败，请确保服务已启动"
    exit 1
fi

# Install dependencies
echo "[2/4] 安装 Python 依赖..."
pip install -r requirements.txt -q

# Start server
echo "[3/4] 初始化数据库并启动服务..."
echo ""
echo "  API 地址: http://0.0.0.0:9000"
echo "  API 文档: http://127.0.0.1:9000/docs"
echo "  管理员: admin / 12345678"
echo ""
echo "[4/4] 服务启动中..."
echo "=========================================="

python run.py
