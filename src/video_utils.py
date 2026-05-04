import os

from moviepy import VideoFileClip


def downscale_video(input_path, output_path, target_width=None, target_height=None, scale_factor=1.0, fps=None):
    """
    使用MoviePy降低视频分辨率

    Args:
        input_path: 输入视频路径
        output_path: 输出视频路径
        target_width: 目标宽度（如果提供，target_height会被自动计算以保持比例）
        target_height: 目标高度
        scale_factor: 缩放因子（0.5表示缩小到一半）
        fps: 目标帧率
    """
    with VideoFileClip(input_path, audio=False) as clip:
        original_width, original_height = clip.w, clip.h
        print(f"视频时长: {clip.duration}s; 原始分辨率: {original_width} × {original_height}")
        if target_width and target_height:
            new_size = (target_width, target_height)
        elif target_width:
            new_height = int(target_width * original_height / original_width)
            new_size = (target_width, new_height)
        elif target_height:
            new_width = int(target_height * original_width / original_height)
            new_size = (new_width, target_height)
        else:
            new_width = int(original_width * scale_factor)
            new_height = int(original_height * scale_factor)
            new_size = (new_width, new_height)
        # 确保新分辨率为偶数（某些编码器要求）
        new_size = (new_size[0] - new_size[0] % 2, new_size[1] - new_size[1] % 2)
        print(f"目标分辨率: {new_size[0]} × {new_size[1]}; 分辨率缩放比例: {new_size[0]/original_width:.2f} 倍")
        # 调整分辨率
        with clip.resized(new_size) as resized_clip:
            # 设置输出视频参数
            output_fps = clip.fps if fps is None else fps
            codec = 'flv' if output_path.endswith(".flv") else 'libx264'  # 常用的视频编码器
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            # 保存调整后的视频
            resized_clip.write_videofile(
                output_path,
                fps=output_fps,
                codec=codec,
                audio_codec='aac' if clip.audio else None
            )
            print(f"视频已保存到: {output_path}")
