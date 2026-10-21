# 接管适应原型

本轮在 `codex/takeover-adaptation` 分支开发，旧分支
`codex/human-takeover-recovery` 已推送 GitHub，旧 CP2 结果及模型不变。
这是借鉴人接管实验协议的自定义 QuadSwarm 任务，并非 HACO、HIL-RL
或上游 QuadSwarm 训练结果的复现。没有移植这些项目的源代码或权重。

## 本轮要验证什么

四架六自由度无人机巡航，其中一架受外部操作者短暂接管。
outward 是向编队外飞行的低冲突对照；crossing 是向编队内穿越的几何条件。
标签仅代表目标航迹几何，具体碰撞难度由规则控制器实测，不能预设。
其他无人机在接管前、接管中和交还后都由同一自主策略控制。
接管者执行独立有界速度指令，接管期间对应 actor 行不参与 PPO 更新；
交还后该无人机立即重新使用自主策略。

三组共享网络架构、初始化种子、终端任务奖励、PPO 设置、动作范围和训练步数：

* blind：当前物理观测，去掉全部显式接管标志、阶段和接管计时。
* flag：当前完整观测，包括接管标志、阶段和已过接管时间。
* history：完整当前观测加前三个控制时刻的本地观测，覆盖 0.3 秒历史。

输入统一为 4×41 维，每组都是共享局部 actor + 独立集中式 critic。
blind/flag 的历史槽位补零；因此总参数量相同，但历史组的信息量和可利用输入权重更多。
这并不能替代后续更严格的记忆结构消融。
所有策略只能看到自身行的观测历史，不能看到隐藏巡检目标、未来指令、
未来接管/交还时刻或操作者风格参数。真实权限掩码仅由环境用于执行覆盖和训练屏蔽。
观测中的显式阶段对 blind 清零，不能通过 actor 的阶段门控把接管事件重新泄漏进去。

模拟操作者包括 steady/delayed/noisy，另有 unseen_slow/unseen_fast 用于开发评估。
延迟以控制间隔为单位，速度噪声每个控制间隔采样并保持，拥有独立可复现 RNG。
不同策略的轨迹会改变操作者反馈，因此配对指相同初态、参数和随机流，
不表示强行重放完全相同的人工动作。

## 服务器工程筛查

路径 `/home/u2025111685/projects/quad-swarm-rl`；Python
`/home/u2025111685/.venvs/quad-swarm/bin/python`。
CPU 单线程运行低维策略和物理仿真；这一步不依赖 GPU，也不测试 GPU 性能。

```bash
cd /home/u2025111685/projects/quad-swarm-rl
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
/home/u2025111685/.venvs/quad-swarm/bin/python -m swarm_rl.takeover.adaptation_train \
  --variant history --seed 0 --output takeover_runs/NEW-history
```

`--smoke` 是 256 步的链路测试，不能与正式 pilot 混合分析。
固定 pilot 每组 16384 步、16 次 PPO 更新、单训练种子；评估 4 个独立初态种子
×2 种几何×5 种风格，共 40 个案例。blind 加 `--include-rules` 时额外评估
nominal/reactive 两个规则。最终模型固定为最后一步，不按评估结果选模型。
输出目录必须全新。默认每个进程 1200 秒墙钟上限；失败保留 metadata 和已有原始数据。
训练未结束的最后一个 episode 单独标记，不算完整任务。

每阶段报告碰撞、最小间距、巡航滞后积分和观测时长；最终同时报告巡检、
交还、恢复和完整任务成功。接管前碰撞也保留。无交还/未恢复/碰撞的恢复代价
统一计 10 秒，原始恢复时长缺失值保留。不能只统计成功案例的恢复均值。
提前碰撞会缩短滞后积分，所以仅在配对双方共同任务成功案例上比较该指标。
人为安排的临时巡检不是策略错误，本轮没有使用“每次接管都惩罚”的奖励。

```bash
python -m swarm_rl.takeover.adaptation_summary \
  --runs takeover_runs/adaptation-pilot-001 --output takeover_runs/NEW-analysis
```

此入口核查来源、同初始权重、训练预算、配对完整性、模型重载和原始数据哈希。
开发案例不可在后续调参后再次宣称是从未接触的最终测试集。
单种子、短训练、固定无障碍场景不支持收敛、新颖性、真实人机协作或投稿结论。

## 真人键盘接管

在服务器启动新会话（也可不加 checkpoint，用 nominal 队友）：

```bash
cd /home/u2025111685/projects/quad-swarm-rl
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/u2025111685/.venvs/quad-swarm/bin/python \
  -m swarm_rl.takeover.keyboard --port 8765 --geometry crossing \
  --checkpoint takeover_runs/adaptation-pilot-001/history/model.pt \
  --output takeover_runs/NEW-human-session
```

在 Mac 另开终端转发，再访问 `http://127.0.0.1:8765`：

```bash
ssh -N -L 8765:127.0.0.1:8765 n1-research
```

服务只绑定服务器回环地址。界面初始暂停，浏览器每 0.1 秒回传键盘状态。
继续后按 T 接管本轮橙色无人机，W/S 改变世界坐标 x，A/D 改变 y，
Q/E 改变 z，G 交还，P 暂停，R 新轮次。橙色十字是操作者可见的巡检目标；
自主 actor 不接收它。XY 和 XZ 两个投影视图用同一物理坐标。
手动接管模式没有自动接管或定时交还，必须由操作者按键触发。
窗口失焦清空按键，0.35 秒无有效指令后速度归零，0.5 秒心跳丢失自动暂停。

轨迹逐控制步保留按键、人工速度、策略动作、请求/实际执行速度、覆盖掩码和状态。
事件日志保留接管/交还/暂停/重置，提前重置的轮次也单独保留为未完成。
为了测量控制延迟，后续正式真人实验还需加入客户端采样与服务端接收时间戳。
自动化接口测试会单独命名为 http-smoke，绝不能当作真人参与数据。
需要操作者真实操作后才能报告人工接管结果；一个操作者只支持可行性预实验。
