# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 13:54:51 2023

@author: furkan.sasmaz
"""

from moviepy.editor import VideoFileClip

def remove_audio(input_path, output_path):
    video_clip = VideoFileClip(input_path)
    video_clip_without_audio = video_clip.set_audio(None)
    video_clip_without_audio.write_videofile(output_path, codec='libx264')

if __name__ == "__main__":
    input_video_path = "araba.mp4"  # Giriş video dosyasının adını buraya yazın
    output_video_path = "araba_sessiz.mp4"  # Çıkış video dosyasının adını buraya yazın

    remove_audio(input_video_path, output_video_path)
