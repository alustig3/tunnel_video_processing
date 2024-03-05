### Dependencies
- [opencv-python](https://github.com/opencv/opencv-python) `pip install opencv-python`
- [matplotlib](https://matplotlib.org/stable/) `pip install matplotlib`
- [numpy](https://numpy.org/) `pip install numpy`
- [ffmpeg](https://ffmpeg.org/)

### Creating plot videos
The annotation file should be a JSON-MIN file exported from Label Studio

run the following from the command line:
`python process_tunnel.py tunnel_fight.mp4 annotations.json`

### Crop and combine the videos into a single output

run the following from the command line:

`ffmpeg -i output/marked.mp4 -i output/plot.mp4 -filter_complex "[0:v]crop=iw:550:0:(ih-550)/2[upper];[upper][1:v]vstack=inputs=2" output/combined.mp4`

