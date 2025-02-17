import os
import soundfile as sf

def convert_flac_to_wav(folder_path):
    # 遍历文件夹中的所有文件
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.flac'):
                # 构建.flac文件的完整路径
                flac_file_path = os.path.join(root, file)
                # 构建对应的.wav文件的完整路径
                wav_file_path = os.path.splitext(flac_file_path)[0] + '.wav'

                try:
                    # 读取.flac文件
                    audio,sr = sf.read(flac_file_path)
                    # 将音频保存为.wav文件
                    sf.write(wav_file_path,audio,sr)
                    print(f"成功将 {flac_file_path} 转换为 {wav_file_path}")
                except Exception as e:
                    print(f"转换 {flac_file_path} 时出错: {e}")

if __name__ == "__main__":
    # 请将这里替换为你实际的文件夹路径
    folder_path = '/mnt/inspurfs/home/wangpengyu/VINP/AudioSamples'
    convert_flac_to_wav(folder_path)
