/**
 * Frontend Reactive State Store (Single Source of Truth)
 */

export const store = {
  models: {
    yolo: { loaded: false },
    segformer: { loaded: false, backbone: "MiT-B2" },
  },
  hasVideo: false,
  isUploading: false,
  isAnalyzing: false,
  videoMeta: null,
  toggles: {
    show_boxes: true,
    show_drivable: true,
    show_lanes: true,
    show_graph: true,
  },
};
