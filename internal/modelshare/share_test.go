package modelshare

import (
	"crypto/rand"
	"os"
	"path/filepath"
	"testing"
)

func writeFakeWeights(t *testing.T, path string, size int) {
	t.Helper()
	buf := make([]byte, size)
	if _, err := rand.Read(buf); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, buf, 0o644); err != nil {
		t.Fatal(err)
	}
}

func TestExportLoadVerifyRoundTrip(t *testing.T) {
	src := filepath.Join(t.TempDir(), "best.pt")
	writeFakeWeights(t, src, 4096)

	out := filepath.Join(t.TempDir(), "bundle")
	m, err := ExportModel(src, out, "alice", "point_model", "week 17")
	if err != nil {
		t.Fatalf("export: %v", err)
	}
	if m.Author != "alice" || m.ModelKind != "point_model" || m.Notes != "week 17" {
		t.Errorf("manifest fields wrong: %+v", m)
	}
	if m.SHA256 == "" || m.SizeBytes != 4096 {
		t.Errorf("checksum/size unset: %+v", m)
	}

	loaded, err := LoadManifest(out)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	if loaded.SHA256 != m.SHA256 {
		t.Errorf("loaded sha mismatch")
	}
	if err := VerifyChecksum(out, loaded); err != nil {
		t.Errorf("verify: %v", err)
	}
}

func TestExport_RefusesNonEmptyDir(t *testing.T) {
	src := filepath.Join(t.TempDir(), "best.pt")
	writeFakeWeights(t, src, 256)
	out := t.TempDir()
	if err := os.WriteFile(filepath.Join(out, "junk.txt"), []byte("hi"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := ExportModel(src, out, "x", "point_model", ""); err == nil {
		t.Error("expected error on non-empty out dir")
	}
}

func TestVerify_DetectsCorruption(t *testing.T) {
	src := filepath.Join(t.TempDir(), "best.pt")
	writeFakeWeights(t, src, 1024)
	out := filepath.Join(t.TempDir(), "bundle")
	m, err := ExportModel(src, out, "a", "point_model", "")
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(out, m.WeightsFilename), []byte("garbage"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := VerifyChecksum(out, m); err == nil {
		t.Error("expected checksum mismatch")
	}
}

func TestCopyWeightsTo_RoundTrips(t *testing.T) {
	src := filepath.Join(t.TempDir(), "best.pt")
	writeFakeWeights(t, src, 2048)
	out := filepath.Join(t.TempDir(), "bundle")
	m, err := ExportModel(src, out, "a", "point_model", "")
	if err != nil {
		t.Fatal(err)
	}
	dst := filepath.Join(t.TempDir(), "restored.pt")
	if err := CopyWeightsTo(out, m, dst); err != nil {
		t.Fatalf("copy: %v", err)
	}
	srcBytes, _ := os.ReadFile(src)
	dstBytes, _ := os.ReadFile(dst)
	if string(srcBytes) != string(dstBytes) {
		t.Error("round-trip bytes differ")
	}
}
