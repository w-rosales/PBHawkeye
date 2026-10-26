"""Local neural model math, safe checkpoints and leakage-free datasets."""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from src.ball_model import (MODEL_FORMAT, NeuralBallLocator, TemporalBallNet,
                            frame_tensor, heatmap_focal_loss, load_ball_model, peak_from_heatmap)
from tools.train_ball_model import PatchDataset, operating_threshold, prepare


def test_temporal_model_output_and_background_loss_are_finite():
    torch.set_num_threads(2)
    model=TemporalBallNet()
    logits=model(torch.zeros(2,9,32,48))
    assert logits.shape==(2,1,16,24)
    target=torch.zeros_like(logits)
    target[0,0,5,7]=1.
    loss=heatmap_focal_loss(logits,target)
    assert torch.isfinite(loss)
    loss.backward()
    assert model.head.weight.grad is not None
    background=heatmap_focal_loss(torch.full((1,1,8,8),-4.),torch.zeros(1,1,8,8))
    assert torch.isfinite(background) and background>0


def test_heatmap_peak_maps_back_to_native_pixel_coordinates():
    heatmap=np.zeros((10,20),np.float32);heatmap[4,7]=.8
    xy,score=peak_from_heatmap(heatmap,(200,100))
    np.testing.assert_allclose(xy,[70.,40.]);assert score==pytest.approx(.8)


def test_court_constraint_excludes_background_peaks_and_validates_dimensions():
    heatmap=np.zeros((10,20),np.float32)
    heatmap[1,1]=.99;heatmap[6,12]=.8
    mask=np.zeros((100,200),np.uint8);mask[40:,80:]=255
    xy,score=peak_from_heatmap(heatmap,(200,100),mask)
    np.testing.assert_allclose(xy,[120.,60.]);assert score==pytest.approx(.8)
    with pytest.raises(ValueError,match="dimensions"):
        peak_from_heatmap(heatmap,(200,100),mask[:10])
    with pytest.raises(ValueError,match="entire image"):
        peak_from_heatmap(heatmap,(200,100),np.zeros_like(mask))


def test_detector_neural_backend_uses_court_and_resets_without_calibration():
    from src.detector import PickleballDetector

    class Locator:
        resets=0
        def reset(self):self.resets+=1
        def update(self,frame,court_mask=None):
            assert court_mask is not None
            return (12.,14.),.98

    detector=PickleballDetector(mode="A")
    locator=Locator();detector.neural_ball=locator
    frame=np.zeros((32,48,3),np.uint8)
    # Neural detections near a player's shoes must not pass through the old veto.
    result=detector._extract_moving_ball(frame,[(0,0,30,20)],None,
                                        court_mask=np.full((32,48),255,np.uint8))
    assert result==((12.,14.),.98)
    assert detector._extract_moving_ball(frame,[],None,court_mask=None)==(None,0.)
    assert locator.resets==1
    detector.reset_ball_history();assert locator.resets==2


def test_frame_tensor_preserves_causal_rgb_order():
    frames=[np.full((8,8,3),(10+i,20+i,30+i),np.uint8) for i in range(3)]
    tensor=frame_tensor(frames)
    assert tensor.shape==(9,8,8)
    assert tensor[0,0,0].item()==pytest.approx(30/127.5-1)
    assert tensor[6,0,0].item()==pytest.approx(32/127.5-1)


def test_checkpoint_roundtrip_and_unqualified_model_cannot_silently_activate(tmp_path):
    model=TemporalBallNet();path=tmp_path/"model.pt"
    torch.save({"format":MODEL_FORMAT,"state_dict":model.state_dict(),"input_width":64,
                "acceptance_threshold":None},path)
    loaded,metadata=load_ball_model(path)
    assert metadata["format"]==MODEL_FORMAT
    for a,b in zip(model.parameters(),loaded.parameters()):
        torch.testing.assert_close(a,b)
    with pytest.raises(ValueError,match="qualifying validation"):
        NeuralBallLocator(path)
    locator=NeuralBallLocator(path,threshold=.99,threads=2)
    frame=np.zeros((48,64,3),np.uint8)
    assert locator.update(frame)==(None,0.)
    assert locator.update(frame)==(None,0.)
    assert locator.update(frame)[0] is None
    locator.reset();assert locator.frames==[]


def test_operating_threshold_is_selected_by_localization_precision():
    records=[{"score":.9-i*.01,"error_px":2.,"inference_ms":1.} for i in range(6)]
    records += [{"score":.5,"error_px":100.,"inference_ms":1.}]
    threshold=operating_threshold(records,.95)
    assert threshold is not None and threshold>.5
    assert operating_threshold(records[:3],.95) is None  # insufficient calibration cases


def test_prepared_temporal_contexts_do_not_overlap_across_splits(tmp_path):
    audit=tmp_path/"audit";(audit/"frames").mkdir(parents=True)
    frames=list(range(150,157))+list(range(243,250))+list(range(273,280))
    labels=[]
    for i in frames:
        image=np.full((48,64,3),40,np.uint8);cv2.circle(image,(10,20),3,(0,240,240),-1)
        cv2.imwrite(str(audit/"frames"/f"{i:06d}.png"),image)
        labels.append({"frame":i,"visibility":"visible","x":10.,"y":20.})
    (audit/"manual_labels.json").write_text(json.dumps({"labels":labels}))
    (audit/"metadata.json").write_text(json.dumps({"video":"fixture.mp4","width":64,"height":48}))
    output=tmp_path/"prepared";prepare(audit,output,width=64)
    manifest=json.loads((output/"manifest.json").read_text())
    contexts={split:{p for s in manifest["samples"] if s["split"]==split for p in s["inputs"]}
              for split in ["train","validation","test"]}
    assert not contexts["train"]&contexts["validation"]
    assert not contexts["validation"]&contexts["test"]
    dataset=PatchDataset(output/"manifest.json",patch_size=32,samples_per_frame=4)
    for i in range(len(dataset)):
        image,target=dataset[i]
        assert image.shape==(9,32,32) and target.shape==(1,16,16)
        assert target.max().item()==(1. if i%4<2 else 0.)
