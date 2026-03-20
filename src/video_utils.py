from moviepy import VideoFileClip


def show_video_infos(video_path):
    try:
        clip = VideoFileClip(video_path)

        # 获取分辨率信息
        width = clip.w
        height = clip.h
        fps = clip.fps
        duration = clip.duration

        print("=" * 50)
        print("视频原始信息:")
        print("=" * 50)
        print(f"文件路径: {video_path}")
        print(f"原始分辨率: {width} × {height}")
        print(f"宽高比: {width/height:.2f}")
        print(f"原始帧率(FPS): {fps:.2f}")
        print(f"视频时长: {duration:.2f}秒 ({duration/60:.2f}分钟)")

        # 判断视频类型
        if width > 1920 or height > 1080:
            resolution_type = "4K或更高"
        elif width >= 1920 or height >= 1080:
            resolution_type = "全高清(1080p)"
        elif width >= 1280 or height >= 720:
            resolution_type = "高清(720p)"
        elif width >= 854 or height >= 480:
            resolution_type = "标清(480p)"
        else:
            resolution_type = "低分辨率"

        print(f"分辨率类型: {resolution_type}")
        print("=" * 50)

        clip.close()
        return {"width": width, "height": height, "fps": fps, "duration": duration}

    except Exception as e:
        print(f"读取视频失败: {e}")
        return None


def downscale_video(input_path, output_path, target_width=None, target_height=None, scale_factor=0.5):
    """
    使用MoviePy降低视频分辨率

    Args:
        input_path: 输入视频路径
        output_path: 输出视频路径
        target_width: 目标宽度（如果提供，target_height会被自动计算以保持比例）
        target_height: 目标高度
        scale_factor: 缩放因子（0.5表示缩小到一半）
    """
    try:
        # 打开原始视频
        clip = VideoFileClip(input_path)

        # 获取原始分辨率
        original_width, original_height = clip.w, clip.h
        print(f"原始分辨率: {original_width} × {original_height}")

        # 计算目标分辨率
        if target_width and target_height:
            # 如果同时指定了宽高，使用指定值
            new_size = (target_width, target_height)
        elif target_width:
            # 只指定宽度，高度按比例计算
            new_height = int(target_width * original_height / original_width)
            new_size = (target_width, new_height)
        elif target_height:
            # 只指定高度，宽度按比例计算
            new_width = int(target_height * original_width / original_height)
            new_size = (new_width, target_height)
        else:
            # 使用缩放因子
            new_width = int(original_width * scale_factor)
            new_height = int(original_height * scale_factor)
            new_size = (new_width, new_height)

        # 确保新分辨率为偶数（某些编码器要求）
        new_size = (new_size[0] - new_size[0] % 2, new_size[1] - new_size[1] % 2)

        print(f"目标分辨率: {new_size[0]} × {new_size[1]}")
        print(f"分辨率缩放比例: {new_size[0]/original_width:.2f} 倍")

        # 调整分辨率
        resized_clip = clip.resized(new_size)

        # 设置输出视频参数
        output_fps = clip.fps
        codec = 'libx264'  # 常用的视频编码器

        # 保存调整后的视频
        resized_clip.write_videofile(
            output_path,
            fps=output_fps,
            codec=codec,
            audio_codec='aac' if clip.audio else None
        )

        print(f"视频已保存到: {output_path}")

        clip.close()
        resized_clip.close()

        return new_size

    except Exception as e:
        print(f"处理视频时出错: {e}")
        return None


def clip_video(input_path: str, output_path: str, end_time: int, start_time: int = 0, codec: str = None):
    """ 按秒剪切视频 """
    video = VideoFileClip(input_path)
    short_video = video.subclipped(start_time, end_time)
    if output_path.endswith(".flv") and codec is None:
        codec = "flv"
    short_video.write_videofile(output_path, codec=codec)
    video.close()
    short_video.close()

