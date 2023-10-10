##### DiffBIR(老旧照片修复)

```shell
# 项目地址
git clone https://github.com/XPixelGroup/DiffBIR.git
# 模型准备
mkdir -p ./weights
wget -P ./weights https://huggingface.co/lxq007/DiffBIR/resolve/main/general_swinir_v1.ckpt
wget -P ./weights https://huggingface.co/lxq007/DiffBIR/resolve/main/general_full_v1.ckpt
wget -P ./weights https://huggingface.co/lxq007/DiffBIR/resolve/main/face_swinir_v1.ckpt
wget -P ./weights https://huggingface.co/lxq007/DiffBIR/resolve/main/face_full_v1.ckpt
# 环境准备(mac下载依赖时失败的可以先注释了重试)
python3.9 -m venv venv
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu
pip install pytorch_lightning==1.4.2
pip install -r requirements.txt
# 保存依赖列表
pip freeze > requirements_mac.txt
# 启动命令
python inference.py \
    --input inputs/demo/general \
    --config configs/model/cldm.yaml \
    --ckpt weights/general_full_v1.ckpt \
    --reload_swinir --swinir_ckpt weights/general_swinir_v1.ckpt \
    --steps 50 \
    --sr_scale 4 \
    --color_fix_type wavelet \
    --output results/demo/general \
    --device mps
```