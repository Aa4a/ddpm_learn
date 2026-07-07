#!/usr/bin/env bash
# 创建 ddpm_learn conda 环境并安装依赖，输出写入 logs/setup.log
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${ROOT}/logs"
LOG_FILE="${LOG_DIR}/setup.log"
ENV_NAME="ddpm_learn"
CONDA_BASE="$(conda info --base)"
PIP="${CONDA_BASE}/envs/${ENV_NAME}/bin/pip"
PYTHON="${CONDA_BASE}/envs/${ENV_NAME}/bin/python"

mkdir -p "${LOG_DIR}"

# 所有输出写入 logs/setup.log（后台运行: nohup bash setup_env.sh &）
exec >> "${LOG_FILE}" 2>&1

echo "========================================"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] 开始安装 ${ENV_NAME}"
echo "日志文件: ${LOG_FILE}"
echo "========================================"

# 网络加速（GitHub / HuggingFace）；pip 大包建议不走代理，见下方注释
if [[ -f /etc/network_turbo ]]; then
  # shellcheck disable=SC1091
  source /etc/network_turbo
  echo "[info] 已启用 network_turbo"
fi

if ! grep -q 'source /etc/network_turbo' ~/.bashrc 2>/dev/null; then
  echo 'source /etc/network_turbo' >> ~/.bashrc
  echo "[info] 已将 network_turbo 写入 ~/.bashrc"
else
  echo "[info] ~/.bashrc 已包含 network_turbo，跳过"
fi

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "[info] conda 环境 ${ENV_NAME} 已存在，跳过创建"
else
  echo "[step 1/3] 创建 conda 环境 (Python 3.10)..."
  conda create -n "${ENV_NAME}" python=3.10 -y
fi

echo "[step 2/3] 安装 PyTorch (CUDA 12.4)..."
# 注意：network_turbo 会拖慢 pip 下载，此处临时关闭代理
if [[ -n "${http_proxy:-}" ]]; then
  unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
  echo "[info] 安装 PyTorch 时临时关闭代理以加速 pip"
fi
export PYTHONUNBUFFERED=1
"${PIP}" install --progress-bar on \
  torch==2.5.1+cu124 torchvision==0.20.1+cu124 \
  --index-url https://download.pytorch.org/whl/cu124

echo "[step 3/3] 安装其余依赖..."
"${PIP}" install --progress-bar on \
  -r "${ROOT}/requirements.txt" \
  -i https://pypi.tuna.tsinghua.edu.cn/simple

echo "[verify] 验证安装..."
"${PYTHON}" - <<'PY'
import torch, torchvision, numpy, matplotlib, tqdm
print(f"torch       : {torch.__version__}  CUDA={torch.cuda.is_available()}")
print(f"torchvision : {torchvision.__version__}")
print(f"numpy       : {numpy.__version__}")
print(f"matplotlib  : {matplotlib.__version__}")
try:
    import diffusers, datasets, accelerate
    print(f"diffusers   : {diffusers.__version__}")
    print(f"datasets    : {datasets.__version__}")
    print(f"accelerate  : {accelerate.__version__}")
except ImportError as e:
    print(f"[warn] HF 包未完全安装: {e}")
PY

echo "========================================"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] 安装完成"
echo "激活环境: conda activate ${ENV_NAME}"
echo "========================================"
