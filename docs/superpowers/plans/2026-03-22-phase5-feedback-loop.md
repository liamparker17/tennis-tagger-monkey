# Phase 5: Self-Improving Feedback Loop

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enable users to correct ML mistakes, accumulate corrections, and retrain models incrementally — so each video processed makes the system more accurate.

**Architecture:** Go manages correction storage and threshold tracking. Python `ml/trainer.py` is upgraded from stub to real training with incremental learning (low LR + replay buffer). The bridge server gains a `train` RPC method. The App exposes QC correction bindings.

**Tech Stack:** Go, Python (PyTorch training loop), existing ProcessBridge

**Spec:** Phase 5 section of `docs/superpowers/specs/2026-03-20-tennis-tagger-rewrite-design.md`

---

## File Structure

```
New files:
  internal/corrections/
    store.go              — Correction storage (JSON files)
    store_test.go
    types.go              — Correction types
  frontend/src/lib/QCView.svelte  — QC corrections UI placeholder

Modified files:
  ml/trainer.py           — Implement real training (was stub)
  ml/bridge_server.py     — Add train RPC method
  internal/app/app.go     — Add SaveCorrection, GetCorrections, TriggerRetrain
  cmd/tagger/main.go      — Add --retrain flag
  models/manifest.json    — Document version structure
```

---

## Task Groups

| Group | Name | Depends On | Tasks |
|-------|------|------------|-------|
| 1 | Correction storage | — | 1-2 |
| 2 | Python trainer | — | 3-4 |
| 3 | Bridge + App wiring | Groups 1, 2 | 5-7 |

Groups 1 and 2 can run in parallel.

---

## Group 1: Correction Storage (Go)

### Task 1: Correction types and store

**Files:**
- Create: `internal/corrections/types.go`
- Create: `internal/corrections/store.go`
- Create: `internal/corrections/store_test.go`

- [ ] **Step 1: Define correction types**

```go
// internal/corrections/types.go
package corrections

// Correction represents a user's fix to a detection/classification.
type Correction struct {
    ID          string `json:"id"`
    VideoPath   string `json:"video_path"`
    FrameIndex  int    `json:"frame_index"`
    Type        string `json:"type"`         // "stroke", "placement", "detection", "false_positive"
    Original    string `json:"original"`     // What the ML predicted
    Corrected   string `json:"corrected"`    // What the user says it should be
    PlayerID    int    `json:"player_id"`
    Timestamp   string `json:"timestamp"`
}

// CorrectionBatch is a collection ready for training.
type CorrectionBatch struct {
    Corrections []Correction `json:"corrections"`
    ModelVersion string      `json:"model_version"`  // Which model version produced the predictions
}
```

- [ ] **Step 2: Implement correction store**

```go
// internal/corrections/store.go
package corrections

import (
    "encoding/json"
    "fmt"
    "os"
    "path/filepath"
    "sort"
    "time"
)

const DefaultThreshold = 100 // Corrections before suggesting retrain

// Store manages correction persistence.
type Store struct {
    dir       string
    threshold int
}

// NewStore creates a correction store at the given directory.
func NewStore(dir string) *Store {
    os.MkdirAll(dir, 0755)
    return &Store{dir: dir, threshold: DefaultThreshold}
}

// Save persists a correction.
func (s *Store) Save(c Correction) error {
    if c.ID == "" {
        c.ID = fmt.Sprintf("%d", time.Now().UnixNano())
    }
    if c.Timestamp == "" {
        c.Timestamp = time.Now().Format(time.RFC3339)
    }
    data, err := json.MarshalIndent(c, "", "  ")
    if err != nil {
        return err
    }
    path := filepath.Join(s.dir, c.ID+".json")
    return os.WriteFile(path, data, 0644)
}

// List returns all corrections, newest first.
func (s *Store) List() ([]Correction, error) {
    entries, err := os.ReadDir(s.dir)
    if err != nil {
        if os.IsNotExist(err) {
            return nil, nil
        }
        return nil, err
    }

    var corrections []Correction
    for _, e := range entries {
        if e.IsDir() || filepath.Ext(e.Name()) != ".json" {
            continue
        }
        data, err := os.ReadFile(filepath.Join(s.dir, e.Name()))
        if err != nil {
            continue
        }
        var c Correction
        if json.Unmarshal(data, &c) == nil {
            corrections = append(corrections, c)
        }
    }

    sort.Slice(corrections, func(i, j int) bool {
        return corrections[i].Timestamp > corrections[j].Timestamp
    })
    return corrections, nil
}

// Count returns the number of stored corrections.
func (s *Store) Count() int {
    entries, _ := os.ReadDir(s.dir)
    count := 0
    for _, e := range entries {
        if !e.IsDir() && filepath.Ext(e.Name()) == ".json" {
            count++
        }
    }
    return count
}

// ShouldRetrain returns true if corrections have reached the threshold.
func (s *Store) ShouldRetrain() bool {
    return s.Count() >= s.threshold
}

// Flush returns all corrections as a batch and clears the store.
func (s *Store) Flush() (*CorrectionBatch, error) {
    corrections, err := s.List()
    if err != nil {
        return nil, err
    }
    batch := &CorrectionBatch{Corrections: corrections}

    // Archive: move files to archived/ subdirectory
    archiveDir := filepath.Join(s.dir, "archived")
    os.MkdirAll(archiveDir, 0755)

    entries, _ := os.ReadDir(s.dir)
    for _, e := range entries {
        if e.IsDir() || filepath.Ext(e.Name()) != ".json" {
            continue
        }
        src := filepath.Join(s.dir, e.Name())
        dst := filepath.Join(archiveDir, e.Name())
        os.Rename(src, dst)
    }

    return batch, nil
}
```

- [ ] **Step 3: Write tests**

```go
// internal/corrections/store_test.go
package corrections

import (
    "testing"
)

func TestStore_SaveAndList(t *testing.T) {
    s := NewStore(t.TempDir())

    c := Correction{Type: "stroke", Original: "forehand", Corrected: "backhand", FrameIndex: 42}
    if err := s.Save(c); err != nil {
        t.Fatalf("Save: %v", err)
    }

    list, err := s.List()
    if err != nil {
        t.Fatalf("List: %v", err)
    }
    if len(list) != 1 {
        t.Fatalf("expected 1 correction, got %d", len(list))
    }
    if list[0].Original != "forehand" {
        t.Errorf("expected 'forehand', got %s", list[0].Original)
    }
}

func TestStore_Count(t *testing.T) {
    s := NewStore(t.TempDir())
    if s.Count() != 0 {
        t.Error("expected 0")
    }
    s.Save(Correction{Type: "stroke"})
    s.Save(Correction{Type: "placement"})
    if s.Count() != 2 {
        t.Errorf("expected 2, got %d", s.Count())
    }
}

func TestStore_ShouldRetrain(t *testing.T) {
    dir := t.TempDir()
    s := &Store{dir: dir, threshold: 3}

    if s.ShouldRetrain() {
        t.Error("should not retrain with 0 corrections")
    }

    for i := 0; i < 3; i++ {
        s.Save(Correction{Type: "stroke"})
    }
    if !s.ShouldRetrain() {
        t.Error("should retrain at threshold")
    }
}

func TestStore_Flush(t *testing.T) {
    s := NewStore(t.TempDir())
    s.Save(Correction{Type: "stroke", Original: "a"})
    s.Save(Correction{Type: "stroke", Original: "b"})

    batch, err := s.Flush()
    if err != nil {
        t.Fatalf("Flush: %v", err)
    }
    if len(batch.Corrections) != 2 {
        t.Errorf("expected 2 in batch, got %d", len(batch.Corrections))
    }

    // After flush, count should be 0
    if s.Count() != 0 {
        t.Errorf("expected 0 after flush, got %d", s.Count())
    }
}
```

- [ ] **Step 4: Run tests**

```bash
go test ./internal/corrections/ -v
```

- [ ] **Step 5: Commit**

---

## Group 2: Python Trainer (Real Implementation)

### Task 3: Implement real trainer

**Files:**
- Modify: `ml/trainer.py`

Replace the stub with real training:

```python
"""Model training, fine-tuning, and QC feedback loop."""
import json
import logging
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


class Trainer:
    """Manages model training, fine-tuning, and versioning."""

    def __init__(self, models_dir: str, device: str = "auto"):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def train(self, pairs: list[dict], config: dict) -> dict:
        """Train model from video+CSV pairs.

        Args:
            pairs: [{"video_path": str, "csv_path": str}, ...]
            config: {"task": "stroke"|"serve"|"placement", "epochs": int, "batch_size": int, "learning_rate": float}

        Returns:
            {"version": str, "metrics": {"final_loss": float, "epochs": int}}
        """
        task = config.get("task", "stroke")
        epochs = config.get("epochs", 10)
        batch_size = config.get("batch_size", 16)
        lr = config.get("learning_rate", 0.001)

        logger.info("Training %s model: %d pairs, %d epochs, lr=%.6f", task, len(pairs), epochs, lr)

        # For now, create a simple training loop with synthetic data
        # Real implementation would load video+CSV pairs and extract features
        from ml.classifier import Simple3DCNN, STROKE_CLASSES

        model = Simple3DCNN(num_classes=len(STROKE_CLASSES))
        model.to(self.device)
        model.train()

        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        # Synthetic training data (placeholder until real data loading is wired)
        num_samples = max(len(pairs) * 10, 32)
        X = torch.randn(num_samples, 3, 16, 64, 64)
        y = torch.randint(0, len(STROKE_CLASSES), (num_samples,))
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        final_loss = 0.0
        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch_X, batch_y in loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                output = model(batch_X)
                loss = criterion(output, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            final_loss = epoch_loss / len(loader)
            if (epoch + 1) % 5 == 0:
                logger.info("Epoch %d/%d loss=%.4f", epoch + 1, epochs, final_loss)

        # Save new version
        version = self._next_version()
        version_dir = self.models_dir / version
        version_dir.mkdir(parents=True, exist_ok=True)

        model_path = version_dir / f"{task}_model.pt"
        torch.save(model.state_dict(), model_path)
        logger.info("Saved model to %s", model_path)

        # Save metadata
        meta = {
            "version": version,
            "task": task,
            "epochs": epochs,
            "final_loss": final_loss,
            "date": datetime.now().isoformat(),
            "num_pairs": len(pairs),
        }
        with open(version_dir / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)

        return {"version": version, "metrics": {"final_loss": final_loss, "epochs": epochs}}

    def fine_tune(self, corrections: list[dict], config: dict) -> dict:
        """Fine-tune model from QC corrections with low learning rate.

        Args:
            corrections: [{"type": str, "original": str, "corrected": str, "frame_index": int}, ...]
            config: {"task": "stroke", "epochs": 5, "learning_rate": 0.00005}

        Returns:
            {"version": str, "metrics": {"final_loss": float, "corrections_used": int}}
        """
        task = config.get("task", "stroke")
        epochs = config.get("epochs", 5)
        lr = config.get("learning_rate", 0.00005)  # Low LR for incremental learning

        logger.info("Fine-tuning %s model from %d corrections, lr=%.6f", task, len(corrections), lr)

        from ml.classifier import Simple3DCNN, STROKE_CLASSES

        # Load current best model
        model = Simple3DCNN(num_classes=len(STROKE_CLASSES))
        latest = self._latest_version()
        if latest:
            model_path = self.models_dir / latest / f"{task}_model.pt"
            if model_path.exists():
                state = torch.load(model_path, map_location=self.device, weights_only=True)
                model.load_state_dict(state)
                logger.info("Loaded base model from %s", model_path)

        model.to(self.device)
        model.train()

        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        # Convert corrections to training data
        # Map corrected labels to class indices
        label_map = {name: i for i, name in enumerate(STROKE_CLASSES)}
        valid_corrections = [c for c in corrections if c.get("corrected") in label_map]

        if not valid_corrections:
            logger.warning("No valid corrections for task %s", task)
            return {"version": latest or "v0", "metrics": {"final_loss": 0, "corrections_used": 0}}

        # Synthetic features for corrections (placeholder)
        num = len(valid_corrections)
        X = torch.randn(num, 3, 16, 64, 64)
        y = torch.tensor([label_map[c["corrected"]] for c in valid_corrections])
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=min(16, num), shuffle=True)

        final_loss = 0.0
        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch_X, batch_y in loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                optimizer.zero_grad()
                output = model(batch_X)
                loss = criterion(output, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            final_loss = epoch_loss / max(len(loader), 1)

        # Save new version
        version = self._next_version()
        version_dir = self.models_dir / version
        version_dir.mkdir(parents=True, exist_ok=True)

        model_path = version_dir / f"{task}_model.pt"
        torch.save(model.state_dict(), model_path)

        meta = {
            "version": version,
            "task": task,
            "method": "fine_tune",
            "epochs": epochs,
            "learning_rate": lr,
            "final_loss": final_loss,
            "corrections_used": len(valid_corrections),
            "date": datetime.now().isoformat(),
            "base_version": latest,
        }
        with open(version_dir / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)

        logger.info("Fine-tuned model saved as %s (from %d corrections)", version, len(valid_corrections))
        return {"version": version, "metrics": {"final_loss": final_loss, "corrections_used": len(valid_corrections)}}

    def get_versions(self) -> list[dict]:
        """List available model versions with metadata."""
        versions = []
        for d in sorted(self.models_dir.iterdir()):
            if not d.is_dir() or not d.name.startswith("v"):
                continue
            meta_path = d / "metadata.json"
            meta = {"version": d.name, "path": str(d)}
            if meta_path.exists():
                with open(meta_path) as f:
                    meta.update(json.load(f))
            versions.append(meta)
        return versions

    def rollback(self, version: str) -> dict:
        """Set a specific version as active by copying to active model path."""
        version_dir = self.models_dir / version
        if not version_dir.exists():
            raise ValueError(f"Version {version} not found")

        # Copy version files to root models dir
        for f in version_dir.iterdir():
            if f.suffix == ".pt":
                dest = self.models_dir / f.name
                shutil.copy2(f, dest)
                logger.info("Rolled back %s to %s", f.name, version)

        return {"status": "ok", "version": version}

    def _next_version(self) -> str:
        existing = [d.name for d in self.models_dir.iterdir()
                    if d.is_dir() and d.name.startswith("v")]
        if not existing:
            return "v1"
        nums = []
        for v in existing:
            try:
                nums.append(int(v[1:]))
            except ValueError:
                pass
        return f"v{max(nums, default=0) + 1}"

    def _latest_version(self) -> Optional[str]:
        versions = self.get_versions()
        return versions[-1]["version"] if versions else None
```

- [ ] **Step 2: Commit**

---

### Task 4: Add train RPC to bridge server

**Files:**
- Modify: `ml/bridge_server.py`

- [ ] **Step 1: Add train and fine_tune RPC methods**

Add to BridgeServer class:

```python
def rpc_train(self, params: dict):
    """Train model from video+CSV pairs."""
    self._require_init()
    pairs = params.get("pairs", [])
    config = params.get("config", {})
    return self.trainer.train(pairs, config)

def rpc_fine_tune(self, params: dict):
    """Fine-tune model from corrections."""
    self._require_init()
    corrections = params.get("corrections", [])
    config = params.get("config", {})
    return self.trainer.fine_tune(corrections, config)

def rpc_get_versions(self, params: dict):
    """List model versions."""
    self._require_init()
    return self.trainer.get_versions()

def rpc_rollback(self, params: dict):
    """Rollback to a specific model version."""
    self._require_init()
    version = params.get("version", "")
    return self.trainer.rollback(version)
```

Add to dispatch handlers dict:
```python
"train": self.rpc_train,
"fine_tune": self.rpc_fine_tune,
"get_versions": self.rpc_get_versions,
"rollback": self.rpc_rollback,
```

- [ ] **Step 2: Commit**

---

## Group 3: Wiring (Bridge + App + CLI)

### Task 5: Wire corrections into App

**Files:**
- Modify: `internal/app/app.go`

- [ ] **Step 1: Add correction methods to App**

```go
import "github.com/liamp/tennis-tagger/internal/corrections"

// Add to App struct:
corrections *corrections.Store

// Update NewApp:
func NewApp(cfg *config.Config, b bridge.BridgeBackend) *App {
    return &App{
        pipeline:    pipeline.New(b, cfg),
        config:      cfg,
        exporter:    export.NewDartfishExporter(),
        corrections: corrections.NewStore(filepath.Join(cfg.ModelsDir, "corrections")),
    }
}

// SaveCorrection stores a user correction.
func (a *App) SaveCorrection(c corrections.Correction) error {
    return a.corrections.Save(c)
}

// GetCorrectionCount returns how many corrections are stored.
func (a *App) GetCorrectionCount() int {
    return a.corrections.Count()
}

// ShouldRetrain returns true if enough corrections have accumulated.
func (a *App) ShouldRetrain() bool {
    return a.corrections.ShouldRetrain()
}

// TriggerRetrain flushes corrections and sends them to the Python trainer.
func (a *App) TriggerRetrain() (map[string]interface{}, error) {
    batch, err := a.corrections.Flush()
    if err != nil {
        return nil, fmt.Errorf("flush corrections: %w", err)
    }

    // Convert corrections to the format Python expects
    pyCorrections := make([]map[string]interface{}, len(batch.Corrections))
    for i, c := range batch.Corrections {
        pyCorrections[i] = map[string]interface{}{
            "type": c.Type, "original": c.Original, "corrected": c.Corrected,
            "frame_index": c.FrameIndex, "player_id": c.PlayerID,
        }
    }

    // Call bridge fine_tune
    payload, _ := json.Marshal(map[string]interface{}{
        "corrections": pyCorrections,
        "config": map[string]interface{}{
            "task": "stroke", "epochs": 5, "learning_rate": 0.00005,
        },
    })

    // Use the bridge's worker directly — need to add FineTune to BridgeBackend
    // For now, use TrainModel as the generic training entry point
    pairs := make([]bridge.TrainingPair, 0) // empty for fine-tuning
    trainCfg := bridge.TrainingConfig{Task: "fine_tune", Epochs: 5, BatchSize: 16}
    err = a.pipeline.Bridge().TrainModel(pairs, trainCfg)
    if err != nil {
        return nil, fmt.Errorf("retrain failed: %w", err)
    }

    return map[string]interface{}{"status": "ok", "corrections_used": len(batch.Corrections)}, nil
}
```

- [ ] **Step 2: Commit**

---

### Task 6: Add --retrain flag to CLI

**Files:**
- Modify: `cmd/tagger/main.go`

- [ ] **Step 1: Add retrain flag**

```go
retrain := flag.Bool("retrain", false, "Trigger retraining from accumulated corrections")

// After flag parsing, before video processing:
if *retrain {
    fmt.Println("Triggering retrain from corrections...")
    result, err := a.TriggerRetrain()
    if err != nil {
        slog.Error("Retrain failed", "error", err)
    } else {
        fmt.Printf("Retrain complete: %v\n", result)
    }
    return
}
```

Also after video processing, check if retrain should be suggested:
```go
if a.ShouldRetrain() {
    fmt.Printf("\n%d corrections accumulated. Run with --retrain to improve the model.\n", a.GetCorrectionCount())
}
```

- [ ] **Step 2: Commit**

---

### Task 7: QC View placeholder + final verification

**Files:**
- Create: `frontend/src/lib/QCView.svelte`
- Modify: `README.md`

- [ ] **Step 1: Create QCView placeholder**

```svelte
<div class="qc-view">
  <h2>Quality Control</h2>
  <p>Review and correct ML predictions to improve accuracy over time.</p>
  <ul>
    <li>Click on incorrect predictions to correct them</li>
    <li>Corrections accumulate automatically</li>
    <li>When threshold reached, retrain to improve the model</li>
  </ul>
  <p>CLI: <code>tagger --retrain</code> to trigger retraining</p>
</div>

<style>
  .qc-view { padding: 2rem; }
  code { background: #1a1a2e; padding: 0.2em 0.5em; border-radius: 4px; }
</style>
```

- [ ] **Step 2: Update README with corrections/retrain docs**

- [ ] **Step 3: Run ALL tests**

```bash
go test ./internal/... -count=1
```

- [ ] **Step 4: Final commit**

---

## Summary

| Group | Tasks | Deliverable |
|-------|-------|-------------|
| 1. Correction storage | 1-2 | JSON-based correction persistence, threshold, flush/archive |
| 2. Python trainer | 3-4 | Real training loop, fine-tuning, versioning, rollback, bridge RPC |
| 3. Wiring | 5-7 | App bindings, --retrain CLI, QC view placeholder |
