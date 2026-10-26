"""Extract a new labeled-data slice and create a local click-to-label page."""

import argparse
import json
from pathlib import Path
import cv2


def extract(video,output,start,count):
    if start<0 or not 1<=count<=3000:
        raise ValueError("Use a nonnegative start and 1-3000 frames per annotation batch")
    cap=cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise ValueError("Could not open the video")
    fps=float(cap.get(cv2.CAP_PROP_FPS));width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH));height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    (output/"frames").mkdir(parents=True,exist_ok=True)
    cap.set(cv2.CAP_PROP_POS_FRAMES,max(0,start-2))
    last=None
    for frame in range(max(0,start-2),start+count):
        ok,image=cap.read()
        if not ok:break
        cv2.imwrite(str(output/"frames"/f"{frame:06d}.png"),image)
        last=frame
    cap.release()
    if last is None or last<start:
        raise ValueError("The requested range has no readable frames")
    metadata={"video":str(video),"fps":fps,"width":width,"height":height,"frame_start":start,"frame_end_inclusive":last}
    (output/"metadata.json").write_text(json.dumps(metadata,indent=2))
    template=Path(__file__).with_name("ball_labeler.html").read_text()
    (output/"label.html").write_text(template.replace("__METADATA__",json.dumps(metadata).replace("</","<\\/")))
    print("Open",output/"label.html","and save exported manual_labels.json in this folder.")


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--video",type=Path,required=True);parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--start",type=int,default=0);parser.add_argument("--count",type=int,default=300)
    args=parser.parse_args();extract(args.video,args.output,args.start,args.count)
