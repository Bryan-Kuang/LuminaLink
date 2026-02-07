# LuminaLink 摄像头模式快速入门

## 概述

LuminaLink现在支持**实时摄像头解说模式**，通过GUI界面为视障用户提供实时电影解说服务。

## 功能特性

✅ **实时摄像头捕获** - 支持任何连接的摄像头设备
✅ **AI场景分析** - 使用GPT-4 Vision理解画面内容
✅ **智能对话检测** - 在角色对话时自动保持沉默
✅ **图形界面** - 易用的Tkinter GUI界面
✅ **实时字幕** - 在画面上叠加显示解说文本
✅ **解说历史** - 记录所有解说内容
✅ **键盘快捷键** - 空格暂停/继续，ESC停止

## 安装依赖

### 1. 安装Python依赖

```bash
pip install -r requirements.txt
```

### 2. 配置API密钥

复制环境变量模板：

```bash
cp .env.example .env
```

编辑`.env`文件，添加你的OpenAI API密钥：

```bash
OPENAI_API_KEY=your-api-key-here
```

## 快速开始

### 基本使用

启动摄像头模式（使用默认摄像头）：

```bash
python -m src.main camera
```

### 指定摄像头

如果你有多个摄像头，可以指定设备索引：

```bash
# 使用第一个摄像头（默认）
python -m src.main camera --camera 0

# 使用第二个摄像头
python -m src.main camera --camera 1
```

### 使用角色识别

如果你想识别特定角色并在解说中使用真实姓名：

```bash
python -m src.main camera --characters data/characters/characters.json
```

角色配置文件格式（`data/characters/characters.json`）：

```json
{
  "characters": [
    {
      "name": "Tony Stark",
      "aliases": ["Iron Man", "Tony"],
      "description": "主角，天才发明家"
    },
    {
      "name": "Peter Parker",
      "aliases": ["Spider-Man", "Peter"],
      "description": "年轻的超级英雄"
    }
  ]
}
```

### 自定义分辨率和帧率

```bash
python -m src.main camera --width 1920 --height 1080 --fps 30
```

## GUI使用指南

### 主界面

启动后会打开GUI窗口，包含以下部分：

```
┌────────────────────────────────────────┐
│     LuminaLink Camera Narration        │
├────────────────────────────────────────┤
│                                        │
│         [摄像头画面预览区]              │
│                                        │
│    [字幕显示在画面底部]                 │
├────────────────────────────────────────┤
│  Camera 0                              │
│  [▶ Start] [⏸ Pause] [⏹ Stop]         │
├────────────────────────────────────────┤
│  Status: Running                       │
│  Frames: 150 | Narrations: 8          │
├────────────────────────────────────────┤
│  Narration Log                         │
│  [12:34:56] 一个男人走进房间            │
│  [12:35:03] 他坐在桌子旁，看起来很专注  │
│  ...                                   │
└────────────────────────────────────────┘
Shortcuts: Space = Pause/Resume | Esc = Stop
```

### 操作步骤

1. **启动应用**：运行`python -m src.main camera`
2. **点击"Start"按钮**：开始捕获摄像头画面
3. **对准屏幕**：将摄像头对准电影屏幕
4. **自动解说**：系统会在对话间隙自动生成并播放解说
5. **查看字幕**：解说文本会显示在画面底部
6. **查看历史**：所有解说记录会显示在日志区域

### 快捷键

| 快捷键 | 功能 |
|--------|------|
| `Space` | 暂停/继续解说 |
| `Esc` | 停止解说 |
| `Ctrl+Q` | 退出应用 |

## 工作原理

```
┌─────────────┐
│ 摄像头输入   │ (30 FPS)
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ 场景分析    │ (1-2 FPS，AI调用)
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ 对话检测    │ (麦克风监听)
└──────┬──────┘
       │
       ▼ (静音期)
┌─────────────┐
│ 生成解说    │ (GPT-4生成文本)
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ TTS播放     │ (EdgeTTS语音合成)
└─────────────┘
```

## 性能优化

### 降低延迟

如果解说延迟过高（>500ms），可以：

1. **降低分辨率**：
   ```bash
   python -m src.main camera --width 640 --height 480
   ```

2. **减少分析频率**：
   编辑`.env`文件：
   ```
   KEYFRAME_INTERVAL=2.0  # 默认1.0秒，改为2秒
   ```

3. **使用更快的AI模型**：
   考虑使用Google Gemini（需要在config中配置）

### 降低API成本

1. **增加解说间隔**：
   ```
   NARRATION_INTERVAL=10  # 默认5秒，改为10秒
   ```

2. **减少分析频率**：参见上述"降低延迟"部分

## 故障排除

### 摄像头无法打开

**错误信息**：
```
Failed to open camera: 0
```

**解决方案**：
1. 检查摄像头是否被其他程序占用
2. 尝试不同的摄像头索引（`--camera 1`, `--camera 2`）
3. 检查系统隐私设置，授予Python摄像头权限
4. macOS：系统偏好设置 → 安全性与隐私 → 摄像头

### 麦克风无法访问

**错误信息**：
```
sounddevice not available
```

**解决方案**：
1. 安装sounddevice：`pip install sounddevice`
2. 检查麦克风权限
3. macOS：系统偏好设置 → 安全性与隐私 → 麦克风

### API调用失败

**错误信息**：
```
OpenAI API error: 401 Unauthorized
```

**解决方案**：
1. 检查`.env`文件中的`OPENAI_API_KEY`是否正确
2. 确认API密钥有效且有余额
3. 检查网络连接

### 延迟过高

**症状**：从画面变化到听到解说超过5秒

**解决方案**：
1. 降低摄像头分辨率（见"性能优化"）
2. 检查网络速度（AI API调用需要网络）
3. 关闭其他占用CPU的程序
4. 考虑使用本地AI模型（未来功能）

## 高级功能

### 多摄像头切换

查看可用摄像头：

```bash
python -c "import cv2; [print(f'Camera {i}') for i in range(5) if cv2.VideoCapture(i).isOpened()]"
```

### 导出解说记录

解说日志会实时显示在GUI中。未来版本将支持导出为文本文件或SRT字幕。

## 与视频文件模式的区别

| 特性 | 视频文件模式 (`process`) | 摄像头模式 (`camera`) |
|------|-------------------------|---------------------|
| 输入源 | 视频文件 | 摄像头实时流 |
| 输出 | SRT字幕文件 | 实时TTS播放 |
| 界面 | 命令行 | GUI窗口 |
| 对话检测 | 从视频音轨 | 从麦克风 |
| 用途 | 批量处理、生成字幕 | 实时观影辅助 |

## 示例场景

### 在家观影

1. 设置摄像头对准电视/投影屏幕
2. 启动LuminaLink：`python -m src.main camera`
3. 点击"Start"开始解说
4. 电影播放过程中自动获得解说

### 在电影院

1. 将手机或便携摄像头对准屏幕
2. 连接耳机以避免打扰他人
3. 启动LuminaLink并开始解说
4. 享受无障碍观影体验

## 系统要求

- **操作系统**：macOS, Windows, Linux
- **Python**：3.8+
- **硬件**：
  - 摄像头（内置或外置）
  - 麦克风（用于对话检测）
  - 扬声器或耳机
  - 建议：4GB+ RAM，多核CPU
- **网络**：稳定的互联网连接（用于AI API调用）

## 隐私说明

- **本地处理**：视频帧在本地捕获和处理
- **AI调用**：仅发送单个帧到OpenAI API进行分析
- **不存储视频**：摄像头画面不会被录制或保存
- **解说记录**：仅在内存中保存，关闭应用后清除

## 反馈与贡献

如果遇到问题或有改进建议，请：

1. 查看日志文件：`~/.luminalink/logs/`
2. 提交Issue：[GitHub Issues](https://github.com/yourusername/LuminaLink/issues)
3. 贡献代码：欢迎Pull Request

## 更新日志

### v1.1.0 (2026-02-06)

- ✨ 新增摄像头实时解说模式
- ✨ 添加GUI界面
- ✨ 支持麦克风对话检测
- ✨ 实时字幕叠加
- ✨ 解说历史记录
- 🐛 修复VideoFrame类型不统一的问题

---

**祝您使用愉快！如有问题，请参考[完整文档](documents/PRD.md)或[功能需求文档](documents/FRD.md)。**
