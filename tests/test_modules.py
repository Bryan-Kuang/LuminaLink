#!/usr/bin/env python
"""
快速测试脚本 - 测试各个模块是否正常工作
"""

import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_imports():
    """测试模块导入"""
    print("测试模块导入...")
    
    try:
        from src.config import get_config, Config
        print("  ✓ config 模块")
    except Exception as e:
        print(f"  ✗ config 模块: {e}")
        return False
    
    try:
        from src.video_processor import VideoProcessor, VideoFrame
        print("  ✓ video_processor 模块")
    except Exception as e:
        print(f"  ✗ video_processor 模块: {e}")
        return False
    
    try:
        from src.audio_detector import AudioDetector, SilenceWindow
        print("  ✓ audio_detector 模块")
    except Exception as e:
        print(f"  ✗ audio_detector 模块: {e}")
        return False
    
    try:
        from src.character_recognizer import CharacterRecognizer, Character
        print("  ✓ character_recognizer 模块")
    except Exception as e:
        print(f"  ✗ character_recognizer 模块: {e}")
        return False
    
    try:
        from src.scene_analyzer import SceneAnalyzer, SceneAnalysis
        print("  ✓ scene_analyzer 模块")
    except Exception as e:
        print(f"  ✗ scene_analyzer 模块: {e}")
        return False
    
    try:
        from src.narrator import Narrator, Narration, NarrationStyle
        print("  ✓ narrator 模块")
    except Exception as e:
        print(f"  ✗ narrator 模块: {e}")
        return False
    
    try:
        from src.tts_engine import TTSManager, TTSResult
        print("  ✓ tts_engine 模块")
    except Exception as e:
        print(f"  ✗ tts_engine 模块: {e}")
        return False
    
    print("\n所有模块导入成功！\n")
    return True


def test_config():
    """测试配置加载"""
    print("测试配置加载...")
    
    from src.config import get_config
    
    config = get_config()
    print(f"  AI 提供商: {config.ai.provider}")
    print(f"  TTS 引擎: {config.tts.engine}")
    print(f"  TTS 语音: {config.tts.voice}")
    print(f"  讲解间隔: {config.narration.interval}秒")
    print(f"  人脸识别阈值: {config.face_recognition.threshold}")
    
    print("\n配置加载成功！\n")
    return True


def test_character_recognizer():
    """测试角色识别器"""
    print("测试角色识别器...")
    
    from src.character_recognizer import CharacterRecognizer
    
    recognizer = CharacterRecognizer()
    
    # 添加测试角色
    char1 = recognizer.add_character(
        name="小明",
        aliases=["主角", "男主"],
        description="电影主人公"
    )
    print(f"  添加角色: {char1.name}")
    
    char2 = recognizer.add_character(
        name="小红",
        aliases=["女主"],
        description="女主人公"
    )
    print(f"  添加角色: {char2.name}")
    
    # 列出角色
    characters = recognizer.list_characters()
    print(f"  角色总数: {len(characters)}")
    
    # 按名称查找
    found = recognizer.get_character_by_name("小明")
    print(f"  按名称查找: {found.name if found else 'None'}")
    
    print("\n角色识别器测试成功！\n")
    return True


def test_narrator():
    """测试讲解生成器"""
    print("测试讲解生成器...")
    
    from src.narrator import Narrator, NarrationStyle
    from src.scene_analyzer import SceneAnalysis
    
    narrator = Narrator(style=NarrationStyle.CONCISE)
    
    # 创建模拟场景分析
    analysis = SceneAnalysis(
        description="小明站在窗前，望着窗外的雨景，眼神中透露出一丝忧愁。",
        actions=["站立", "凝视"],
        emotions={"小明": "忧愁"},
        setting="室内，窗边",
        objects=["窗户", "雨"],
        timestamp=10.0,
        confidence=0.9
    )
    
    # 生成讲解
    narration = narrator.generate_narration(
        analysis,
        slot=(10.0, 15.0),
        characters_in_frame=["小明"]
    )
    
    if narration:
        print(f"  生成讲解: {narration.text}")
        print(f"  时间范围: {narration.start_time}s - {narration.end_time}s")
    
    print("\n讲解生成器测试成功！\n")
    return True


async def test_tts():
    """测试语音合成"""
    print("测试语音合成...")
    
    try:
        from src.tts_engine import TTSManager
        
        tts = TTSManager(engine_type="edge")
        
        result = await tts.synthesize("你好，这是一个测试。")
        
        if result.success:
            print(f"  合成成功: {result.audio_path}")
            print(f"  音频时长: {result.duration}秒")
            
            # 清理测试文件
            import os
            if os.path.exists(result.audio_path):
                os.remove(result.audio_path)
                print("  已清理测试文件")
        else:
            print(f"  合成失败: {result.error}")
            return False
        
        print("\n语音合成测试成功！\n")
        return True
        
    except ImportError as e:
        print(f"  跳过 (缺少依赖): {e}")
        return True


def main():
    """运行所有测试"""
    print("=" * 50)
    print("LuminaLink 模块测试")
    print("=" * 50 + "\n")
    
    results = []
    
    # 测试导入
    results.append(("模块导入", test_imports()))
    
    # 测试配置
    results.append(("配置加载", test_config()))
    
    # 测试角色识别器
    results.append(("角色识别器", test_character_recognizer()))
    
    # 测试讲解生成器
    results.append(("讲解生成器", test_narrator()))
    
    # 测试 TTS
    import asyncio
    results.append(("语音合成", asyncio.run(test_tts())))
    
    # 汇总
    print("=" * 50)
    print("测试结果汇总")
    print("=" * 50)
    
    all_passed = True
    for name, passed in results:
        status = "✓ 通过" if passed else "✗ 失败"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False
    
    print()
    if all_passed:
        print("🎉 所有测试通过！")
    else:
        print("❌ 部分测试失败，请检查错误信息")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
