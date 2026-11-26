/**
 * Component Thanh Điều Hướng Trên Cùng (Top Header Status Pills)
 */

const dotYolo = document.getElementById("dot-yolo");
const labelYolo = document.getElementById("label-yolo");
const dotSeg = document.getElementById("dot-segformer");
const labelSeg = document.getElementById("label-segformer");

export function renderHeaderStatus(models) {
  const yoloLoaded = Boolean(models?.yolo?.loaded);
  const segLoaded = Boolean(models?.segformer?.loaded);

  if (dotYolo && labelYolo) {
    dotYolo.className = yoloLoaded ? "dot-green" : "dot-amber";
    labelYolo.textContent = yoloLoaded ? "YOLOv12-L" : "YOLO Standby";
  }

  if (dotSeg && labelSeg) {
    dotSeg.className = segLoaded ? "dot-green" : "dot-amber";
    labelSeg.textContent = segLoaded ? "SegFormer MiT-B2" : "SegFormer Standby";
  }
}
