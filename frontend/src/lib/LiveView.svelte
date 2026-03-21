<script>
  // LiveView placeholder - will be wired to Go backend via Wails bindings.
  // Displays live detection results as they stream in from ProcessLive.

  let detectionCount = 0;
  let frameCount = 0;
  let isLive = false;

  async function startLive() {
    // TODO: Call Go StartLive binding via Wails runtime
    isLive = true;
  }

  async function stopLive() {
    // TODO: Call Go StopLive binding via Wails runtime
    isLive = false;
  }
</script>

<div class="live-view">
  <h2>Live Tagging</h2>

  {#if isLive}
    <div class="stats">
      <p>Frames processed: {frameCount}</p>
      <p>Detections: {detectionCount}</p>
    </div>
    <button on:click={stopLive}>Stop</button>
  {:else}
    <button on:click={startLive}>Start Live</button>
  {/if}
</div>

<style>
  .live-view {
    padding: 1rem;
  }
  .stats {
    margin: 1rem 0;
  }
</style>
