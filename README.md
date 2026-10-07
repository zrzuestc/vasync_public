# VA_Sync

vasync 是用于逐步实现同题辩论、价值预测和同步策略实验的工程仓库。

## 大模型配置

所有可选的大模型 API 配置统一写在：

```text
configs/llm_api_configs.json
```

该文件中包含一个 `default` 字段和一个 `profiles` 字段。`default` 表示默认使用哪个大模型配置；`profiles` 下的每个 key 是一个可选的大模型配置名，例如：

```text
qwen3_6_35b_a3b_local
qwen3_5_27b_local
glm_4_7_flash
```

实验配置中可以通过 `llm_profile` 指定默认使用哪个模型：

```json
"llm_profile": "default"
```

运行脚本时，也可以通过命令行临时切换模型：

```bash
python experiments/profile_debate_probe.py \
  --config configs/profile_debate_probe.json \
  --llm-profile glm_4_7_flash \
  --sample-count 1
```

如果不传 `--llm-profile`，脚本会优先使用实验配置中的 `llm_profile`；如果实验配置中也没有写，则使用 `configs/llm_api_configs.json` 里的 `default`。旧的 `qwen3_5_27b_local` profile 暂时保留为兼容别名，实际指向当前本地 `qwen3.6-35b-a3b` 服务。

## 运行 Profile Debate Baseline

在 `vasync` conda 环境中，跑 profile-aware 无同步多智能体 Debate baseline：

```bash
python experiments/profile_debate_probe.py \
  --config configs/profile_debate_probe.json \
  --sample-count 3
```

只跑部分数据集或临时切换模型：

```bash
python experiments/profile_debate_probe.py \
  --config configs/profile_debate_probe.json \
  --llm-profile glm_4_7_flash \
  --datasets gsm8k,mmlu \
  --sample-count 3
```

默认输出目录：

```text
outputs/profile_debate_probe/
```

查看批次 summary：

```bash
python -m json.tool outputs/profile_debate_probe/summary.json
```

查看第一道题完整日志：

```bash
python -c "import json;print(json.dumps(json.loads(open('outputs/profile_debate_probe/questions.jsonl').readline()),ensure_ascii=False,indent=2))"
```
