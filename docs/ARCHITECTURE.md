# 系统架构（一期：视觉转文字）

## 1. 总体数据流

输入：视频文件/视频流 → 解码 → 帧/音频分发 → 视觉理解与对话检测 → 解说生成 → 时间对齐 → 输出。

## 2. 核心设计原则

- 单一职责：每个模块只做一件事（检测/识别/生成/对齐/输出）。
- 可插拔：模型实现可替换（不同推理后端/不同训练版本）。
- 时间优先：所有中间产物必须携带统一时间戳（ms）。
- 流式处理：使用队列与滑动窗口，控制延迟与吞吐。

## 3. 模块划分（建议接口）

- `InputAdapter`：输出 `VideoFrame`（image + pts_ms）与 `AudioChunk`（pcm + start_ms/end_ms）。
- `SceneChangeDetector`：输入帧序列，输出 `SceneChangeEvent`/`SceneSegment`。
- `ObjectDetector`（PyTorch）：输入帧，输出对象/人物检测框与类别。
- `FacePipeline`（PyTorch）：人脸检测→对齐→embedding→与 `CharacterStore` 比对。
- `DialogueDetector`：输入音频chunk，输出对话区间或实时状态。
- `NarrationGenerator`（TensorFlow）：输入结构化 `VisualFacts` 与 `DialogueState`，输出 `NarrationCandidate`。
- `Synchronizer`：合并各事件流，保证时间一致性，并按规则“对话期间禁发”。
- `OutputSink`：将 `NarrationSegment` 输出到 JSONL/回调/消息队列，二期 TTS 作为另一个 sink。

## 4. 性能与实时性策略（一期建议）

- 解码层：优先使用硬件解码（若可用），输出BGR/RGB帧。
- 采样层：对视觉模型可采用“关键帧+低频刷新”的策略；对话检测用连续音频更稳。
- 并行层：解码、视觉推理、音频VAD、NLG分线程/进程，使用有界队列避免堆积。
- 延迟控制：统一 `max_end_to_end_latency_ms` 与回压策略。

## 5. 时间同步

- 以视频PTS为主或以音频时钟为主（需固定一种），所有输出事件都映射到该时间轴。
- 对话检测产生的区间用于 gate 解说输出（允许在对话结束后生成总结性解说，作为可选功能）。
