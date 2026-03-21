package corrections

import (
	"os"
	"testing"
	"time"
)

func TestStore_SaveAndList(t *testing.T) {
	dir := t.TempDir()
	store := NewStore(dir)

	c := Correction{
		VideoPath:  "match.mp4",
		FrameIndex: 42,
		Type:       "stroke",
		Original:   "forehand",
		Corrected:  "backhand",
		PlayerID:   1,
	}

	if err := store.Save(c); err != nil {
		t.Fatalf("Save: %v", err)
	}

	list, err := store.List()
	if err != nil {
		t.Fatalf("List: %v", err)
	}
	if len(list) != 1 {
		t.Fatalf("expected 1 correction, got %d", len(list))
	}

	got := list[0]
	if got.ID == "" {
		t.Error("expected auto-generated ID")
	}
	if got.Timestamp == "" {
		t.Error("expected auto-generated Timestamp")
	}
	if got.VideoPath != "match.mp4" {
		t.Errorf("VideoPath = %q, want %q", got.VideoPath, "match.mp4")
	}
	if got.FrameIndex != 42 {
		t.Errorf("FrameIndex = %d, want 42", got.FrameIndex)
	}
	if got.Type != "stroke" {
		t.Errorf("Type = %q, want %q", got.Type, "stroke")
	}
	if got.Original != "forehand" {
		t.Errorf("Original = %q, want %q", got.Original, "forehand")
	}
	if got.Corrected != "backhand" {
		t.Errorf("Corrected = %q, want %q", got.Corrected, "backhand")
	}
	if got.PlayerID != 1 {
		t.Errorf("PlayerID = %d, want 1", got.PlayerID)
	}
}

func TestStore_Count(t *testing.T) {
	dir := t.TempDir()
	store := NewStore(dir)

	if n := store.Count(); n != 0 {
		t.Fatalf("initial Count = %d, want 0", n)
	}

	for i := 0; i < 2; i++ {
		if err := store.Save(Correction{Type: "stroke"}); err != nil {
			t.Fatalf("Save #%d: %v", i, err)
		}
		// Small sleep to ensure distinct UnixNano IDs on fast machines.
		time.Sleep(time.Millisecond)
	}

	if n := store.Count(); n != 2 {
		t.Fatalf("Count after 2 saves = %d, want 2", n)
	}
}

func TestStore_ShouldRetrain(t *testing.T) {
	dir := t.TempDir()
	store := NewStore(dir)
	store.SetThreshold(3)

	for i := 0; i < 2; i++ {
		_ = store.Save(Correction{Type: "stroke"})
		time.Sleep(time.Millisecond)
	}

	if store.ShouldRetrain() {
		t.Fatal("ShouldRetrain should be false at count=2, threshold=3")
	}

	_ = store.Save(Correction{Type: "stroke"})

	if !store.ShouldRetrain() {
		t.Fatal("ShouldRetrain should be true at count=3, threshold=3")
	}
}

func TestStore_Flush(t *testing.T) {
	dir := t.TempDir()
	store := NewStore(dir)

	for i := 0; i < 2; i++ {
		if err := store.Save(Correction{Type: "placement"}); err != nil {
			t.Fatalf("Save #%d: %v", i, err)
		}
		time.Sleep(time.Millisecond)
	}

	batch, err := store.Flush()
	if err != nil {
		t.Fatalf("Flush: %v", err)
	}
	if len(batch.Corrections) != 2 {
		t.Fatalf("batch has %d corrections, want 2", len(batch.Corrections))
	}

	if n := store.Count(); n != 0 {
		t.Fatalf("Count after flush = %d, want 0", n)
	}

	// Verify archived files exist.
	archived, err := os.ReadDir(dir + "/archived")
	if err != nil {
		t.Fatalf("read archived dir: %v", err)
	}
	if len(archived) != 2 {
		t.Fatalf("archived has %d files, want 2", len(archived))
	}
}
