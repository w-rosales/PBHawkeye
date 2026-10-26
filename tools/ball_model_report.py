"""Render a frozen neural-vs-classical comparison without running inference."""

import argparse
import html
import json
from pathlib import Path

import cv2
import numpy as np


def tracking_stats(rows):
    correct=sum(row["classical_error_px"] is not None and row["classical_error_px"]<=10 for row in rows)
    missing=sum(row["classical"] is None for row in rows)
    return {"visible_frames":len(rows),"correct_within_10px":correct,"wrong_accepted":len(rows)-correct-missing,
            "missing":missing,"recall":correct/max(1,len(rows))}


def render(evaluation_path, primary_path, external_path, output):
    result=json.loads(evaluation_path.read_text())
    threshold=result["threshold"]
    if threshold is None:
        raise ValueError("The model has no validation-selected operating threshold")
    sources={"test":json.loads(primary_path.read_text()),"external":json.loads(external_path.read_text())}
    output.mkdir(parents=True,exist_ok=True)
    rows=[]
    for split in ["test","external"]:
        classical={r["frame"]:r for r in sources[split]["records"]}
        for prediction in result["records"][split]:
            if prediction["frame"] not in classical:
                raise ValueError("Missing classical comparison frame")
            record=classical[prediction["frame"]]
            point=record["measurement"]
            rows.append({**prediction,"split":split,"neural_accepted":prediction["score"]>=threshold,
                         "classical":point,"classical_error_px":float(np.linalg.norm(np.asarray(point)-prediction["target"])) if point is not None else None})
    summary={split:{"classical":tracking_stats([r for r in rows if r["split"]==split]),
                    "neural":result["splits"][split]["validation_threshold"]} for split in ["test","external"]}
    (output/"comparison.json").write_text(json.dumps({"summary":summary,"threshold":threshold,"rows":rows},indent=2))
    video=output/"comparison.mp4"
    writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*"mp4v"),10.,(1920,620))
    if not writer.isOpened():
        raise RuntimeError("Could not create comparison MP4")
    for index,row in enumerate(rows):
        raw=cv2.imread(row["raw_frame"])
        if raw is None:raise ValueError("Missing raw comparison frame")
        target=tuple(round(p) for p in row["target"])
        panels=[]
        for title,point,color,status in [
            ("Reviewed reference",row["target"],(255,255,0),"Approximate 2D center"),
            ("Current classical detector",row["classical"],(0,180,255),"Missing" if row["classical_error_px"] is None else f'Error {row["classical_error_px"]:.1f} px'),
            ("Local neural model",row["predicted"] if row["neural_accepted"] else None,(230,80,255),f'Score {row["score"]:.3f}; '+(f'error {row["error_px"]:.1f} px' if row["neural_accepted"] else "withheld"))]:
            annotated=raw.copy()
            cv2.circle(annotated,target,10,(255,255,0),2)
            if point is not None:
                xy=tuple(round(p) for p in point)
                cv2.circle(annotated,xy,22,color,3)
                cv2.drawMarker(annotated,xy,color,cv2.MARKER_CROSS,18,2)
            canvas=np.full((620,640,3),(25,20,15),np.uint8)
            canvas[48:408]=cv2.resize(annotated,(640,360),interpolation=cv2.INTER_AREA)
            cv2.putText(canvas,title,(16,30),0,.72,color,2)
            cv2.putText(canvas,status,(16,435),0,.6,color,1)
            x=max(0,min(raw.shape[1]-100,target[0]-50));y=max(0,min(raw.shape[0]-80,target[1]-40))
            crop=annotated[y:y+80,x:x+100]
            canvas[450:610,16:216]=cv2.resize(crop,(200,160),interpolation=cv2.INTER_NEAREST)
            cv2.putText(canvas,f'{row["split"]} F{row["frame"]}',(235,485),0,.7,(220,220,220),1)
            cv2.putText(canvas,"Same raw frame",(235,520),0,.55,(220,220,220),1)
            cv2.putText(canvas,"Reference crop at left",(235,550),0,.5,(220,220,220),1)
            panels.append(canvas)
        frame=cv2.hconcat(panels)
        if index==0:cv2.imwrite(str(output/"preview.jpg"),frame)
        for _ in range(3):writer.write(frame)  # Inspection montage, explicitly not real-time playback.
    writer.release()
    table=[]
    for split,stats in summary.items():
        for name,values in stats.items():
            table.append(f'<tr><td>{split}</td><td>{name}</td><td>{values["visible_frames"]}</td><td>{values["correct_within_10px"]}</td><td>{values["wrong_accepted"]}</td><td>{values["missing"]}</td></tr>')
    page='''<!doctype html><meta charset="utf-8"><title>Local ball model experiment</title>
<style>body{max-width:1200px;margin:32px auto;background:#131c28;color:#e4edf7;font:16px system-ui;line-height:1.6;padding:0 20px}a{color:#9cd8ff}video,img{width:100%}table{border-collapse:collapse}td,th{padding:10px;border:1px solid #526076;text-align:left}</style>
<h1>Local ball model: held-out comparison</h1>
<p>A three-frame convolutional heatmap network trained locally on 79 labeled frames. Checkpoint selected on 25 separate validation frames; test inputs did not overlap training inputs. The later clip is from the same match and camera. This is a small experiment, not proof of accuracy on other courts.</p>
<p>Correct means within 10 native 1920×1080 pixels of an approximate reference created through assistant visual review and local RGB center snapping. Independent human validation is still needed. Partial/ambiguous frames are excluded. Precision on these visible frames does not measure false detections when no ball is visible. Compare detector measurements, before Kalman filtering.</p>
<table><tr><th>Split</th><th>Detector</th><th>Visible frames</th><th>Correct</th><th>Wrong accepted</th><th>Missing / withheld</th></tr>'''+''.join(table)+'''</table>
<p>Neural score threshold: '''+html.escape(f'{threshold:.6f}')+''', chosen using validation only and then frozen. Scores are not calibrated probabilities.</p>
<video controls preload="metadata" poster="preview.jpg" src="comparison.mp4"></video>
<p>Slow inspection montage: three repeated display frames per evaluated source frame, not real-time footage. Cyan shows the reference; orange shows the current detector; magenta shows accepted neural detections. Rejected neural peaks are withheld.</p>
<p>This measures 2D localization only. It does not establish 3D ball position, height or MPH. No independent speed measurements exist in this clip. The default Studio tracker has not been switched to this experimental model.</p>
<p><a href="comparison.json">Comparison records</a> · <a href="evaluation.json">Frozen model evaluation</a></p>'''
    (output/"index.html").write_text(page,encoding="utf-8")
    (output/"evaluation.json").write_bytes(evaluation_path.read_bytes())
    print(json.dumps(summary,indent=2))
    print("Review:",output/"index.html")


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--evaluation",type=Path,default=Path("data/models/ball_v1/evaluation_court.json"))
    parser.add_argument("--classical-primary",type=Path,default=Path("data/validation/gold_ball_audit/candidate_warm.json"))
    parser.add_argument("--classical-external",type=Path,default=Path("data/validation/gold_ball_holdout/classical.json"))
    parser.add_argument("--output",type=Path,default=Path("data/validation/ball_model_experiment"))
    args=parser.parse_args();render(args.evaluation,args.classical_primary,args.classical_external,args.output)
