# SyncNet

## Dependencies
```
pip install -r requirements.txt
```

In addition, `ffmpeg` is required.

## run syncnet
```
python syncnet.py --videofile single_mp4_filename --data_dir output_dir --keep_output (it is a flag arg if you want to keep intermediate data)
```

data_dir will include intermediate data, syncnet_results.json(include average confidence score, min distance, and the axis of best dist score in matrix activated.pckl, and the confidence score for all segments), activesd.pckl (include a matrxi of dist scores. dim is number 0f segment * [-15,15] for 15 frames around the segment)

If you want to run a function, then import run_syncnet_pipeline in syncnet.py and run it with your mp4 file.

Below if the original demo from the repo.

## Demo

SyncNet demo:
```
python demo_syncnet.py --videofile data/example.avi --tmp_dir /path/to/temp/directory
```

Check that this script returns:
```
AV offset:      3 
Min dist:       5.353
Confidence:     10.021
```

Full pipeline:
```
sh download_model.sh
python run_pipeline.py --videofile /path/to/video.mp4 --reference name_of_video --data_dir /path/to/output
python run_syncnet.py --videofile /path/to/video.mp4 --reference name_of_video --data_dir /path/to/output
python run_visualise.py --videofile /path/to/video.mp4 --reference name_of_video --data_dir /path/to/output
```

Outputs:
```
$DATA_DIR/pycrop/$REFERENCE/*.avi - cropped face tracks
$DATA_DIR/pywork/$REFERENCE/offsets.txt - audio-video offset values
$DATA_DIR/pyavi/$REFERENCE/video_out.avi - output video (as shown below)
```
<p align="center">
  <img src="img/ex1.jpg" width="45%"/>
  <img src="img/ex2.jpg" width="45%"/>
</p>

## Publications
 
```
@InProceedings{Chung16a,
  author       = "Chung, J.~S. and Zisserman, A.",
  title        = "Out of time: automated lip sync in the wild",
  booktitle    = "Workshop on Multi-view Lip-reading, ACCV",
  year         = "2016",
}
```
