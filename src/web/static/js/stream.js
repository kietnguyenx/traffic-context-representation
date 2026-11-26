/**
 * WebSocket Binary + JSON Stream Client (High-FPS Zero-Lag)
 * Nhận trực tiếp khung hình JPEG nhị phân (ArrayBuffer -> Blob) không qua Base64.
 */

export class AnalysisStreamClient {
  constructor(handlers) {
    this.handlers = handlers;
    this.ws = null;
  }

  start() {
    this.close();
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const wsUrl = `${protocol}//${window.location.host}/ws/analyze`;

    this.ws = new WebSocket(wsUrl);
    this.ws.binaryType = "blob";

    this.ws.onopen = () => {
      if (this.handlers.onOpen) this.handlers.onOpen();
    };

    this.ws.onmessage = (event) => {
      if (event.data instanceof Blob) {
        if (this.handlers.onFrameBlob) this.handlers.onFrameBlob(event.data);
        return;
      }
      try {
        const msg = JSON.parse(event.data);
        if (msg.type === "telemetry" && this.handlers.onTelemetry) {
          this.handlers.onTelemetry(msg);
        } else if (msg.type === "completed" && this.handlers.onCompleted) {
          this.handlers.onCompleted(msg);
        } else if (msg.type === "cancelled" && this.handlers.onCancelled) {
          this.handlers.onCancelled();
        } else if (msg.type === "error" && this.handlers.onError) {
          this.handlers.onError(msg.message || "Error occurred during video analysis.");
        }
      } catch (err) {
        console.error("[WS Parse Error]", err);
      }
    };

    this.ws.onerror = () => {
      if (this.handlers.onError) {
        this.handlers.onError("WebSocket connection to AI analysis stream lost.");
      }
    };

    this.ws.onclose = () => {
      this.ws = null;
      if (this.handlers.onClose) this.handlers.onClose();
    };
  }

  sendToggles(toggles) {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ type: "toggles", data: toggles }));
    }
  }

  stop() {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ type: "stop" }));
    }
  }

  close() {
    if (this.ws) {
      try {
        this.ws.close();
      } catch (_) {
        // ignore
      }
      this.ws = null;
    }
  }
}
