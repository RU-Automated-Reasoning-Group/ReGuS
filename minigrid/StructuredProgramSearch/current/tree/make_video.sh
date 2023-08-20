echo y | ffmpeg -f image2 -framerate 30 -i frames/img%6d.png intermediate.mp4 # ; echo y | ffmpeg -i intermediate.mp4 -vcodec libx264 -pix_fmt yuv420p test_without_move_to_block_close.mp4 
