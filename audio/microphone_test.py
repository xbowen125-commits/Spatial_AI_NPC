"""枚举并测试麦克风；只显示 RMS，不保存音频。"""

import queue
import time

import numpy as np

from audio_config import BLOCK_SIZE, CHANNELS, MIC_DEVICE, SAMPLE_RATE


def main():
    try:
        import msvcrt
        import sounddevice as sd
    except ImportError as error:
        print(f"缺少音频依赖：{error}")
        print("请运行：python -m pip install sounddevice")
        return

    print("可用音频设备：")
    print(sd.query_devices())
    default_input = sd.default.device[0]
    print(f"\n默认输入设备编号：{default_input}")
    print(f"当前配置设备：{MIC_DEVICE if MIC_DEVICE is not None else '系统默认'}")
    print("开始显示麦克风 RMS；按 q 或 Ctrl+C 退出。\n")

    rms_queue = queue.Queue(maxsize=4)

    def callback(indata, frames, time_info, status):
        if status:
            print(f"\n音频警告：{status}")
        samples = np.asarray(indata[:, 0], dtype=np.float32)
        rms = float(np.sqrt(np.mean(samples * samples))) if samples.size else 0.0
        try:
            rms_queue.put_nowait(rms)
        except queue.Full:
            pass

    try:
        with sd.InputStream(
            device=MIC_DEVICE,
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            blocksize=BLOCK_SIZE,
            dtype="float32",
            callback=callback,
        ):
            while True:
                try:
                    rms = rms_queue.get(timeout=0.1)
                    bar = "#" * min(50, int(rms * 500))
                    print(f"\rRMS: {rms:0.4f} {bar:<50}", end="", flush=True)
                except queue.Empty:
                    pass
                if msvcrt.kbhit() and msvcrt.getwch().lower() == "q":
                    break
                time.sleep(0.01)
    except KeyboardInterrupt:
        pass
    except Exception as error:
        print(f"\n无法读取麦克风：{error}")
    finally:
        print("\n麦克风测试已结束。")


if __name__ == "__main__":
    main()
