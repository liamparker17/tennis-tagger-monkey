package modelshare

import (
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

const (
	machineIDFilename = "machine-id"
	syncStateFilename = ".sync-state.json"
)

// MachineID returns a stable, machine-local identifier. Created on first
// call as 16 random bytes hex-encoded, stored at ~/.tennis-tagger/machine-id.
// Used so the same machine's published bundles can be filtered out when
// syncing from a shared folder both machines write into.
func MachineID() (string, error) {
	home, err := os.UserHomeDir()
	if err != nil {
		return "", err
	}
	dir := filepath.Join(home, ".tennis-tagger")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return "", err
	}
	path := filepath.Join(dir, machineIDFilename)
	if b, err := os.ReadFile(path); err == nil {
		id := strings.TrimSpace(string(b))
		if len(id) >= 8 {
			return id, nil
		}
	}
	buf := make([]byte, 16)
	if _, err := rand.Read(buf); err != nil {
		return "", err
	}
	id := hex.EncodeToString(buf)
	if err := os.WriteFile(path, []byte(id), 0o600); err != nil {
		return "", err
	}
	return id, nil
}

// SyncState tracks the latest bundle timestamp seen per remote machine, so
// repeated sync runs don't re-merge bundles that have already been folded
// in. Stored next to the local best.pt as `.sync-state.json`.
type SyncState struct {
	// LastSeen[machineID] = RFC3339 timestamp of the newest bundle we've
	// already merged from that machine.
	LastSeen map[string]string `json:"last_seen"`
}

func LoadSyncState(localPt string) *SyncState {
	s := &SyncState{LastSeen: map[string]string{}}
	data, err := os.ReadFile(filepath.Join(filepath.Dir(localPt), syncStateFilename))
	if err != nil {
		return s
	}
	_ = json.Unmarshal(data, s)
	if s.LastSeen == nil {
		s.LastSeen = map[string]string{}
	}
	return s
}

func (s *SyncState) Save(localPt string) error {
	data, _ := json.MarshalIndent(s, "", "  ")
	return os.WriteFile(filepath.Join(filepath.Dir(localPt), syncStateFilename),
		data, 0o644)
}

// PublishBundle writes the local model to <sharedRoot>/<machineID>/<ts>/
// using the same Manifest format as ExportModel. Returns the full bundle
// directory path.
func PublishBundle(srcPt, sharedRoot, author, notes string) (string, *Manifest, error) {
	id, err := MachineID()
	if err != nil {
		return "", nil, fmt.Errorf("machine id: %w", err)
	}
	// Filesystem-safe timestamp (Windows hates colons).
	ts := time.Now().UTC().Format("2006-01-02T150405Z")
	bundleDir := filepath.Join(sharedRoot, id, ts)
	if err := os.MkdirAll(filepath.Dir(bundleDir), 0o755); err != nil {
		return "", nil, fmt.Errorf("create machine dir: %w", err)
	}
	if entries, _ := os.ReadDir(bundleDir); len(entries) > 0 {
		return "", nil, fmt.Errorf("bundle dir %q already exists and is non-empty", bundleDir)
	}
	if err := os.MkdirAll(bundleDir, 0o755); err != nil {
		return "", nil, err
	}
	m, err := exportInto(srcPt, bundleDir, author, "point_model", notes)
	if err != nil {
		return "", nil, err
	}
	return bundleDir, m, nil
}

// exportInto is ExportModel without the empty-dir check (we just created
// the dir, and pre-checked emptiness above).
func exportInto(srcPt, outDir, author, modelKind, notes string) (*Manifest, error) {
	// Reuse ExportModel by writing into a sibling temp dir and renaming —
	// simplest way to not duplicate the hash/copy logic. ExportModel asserts
	// the dir is empty, so we point it at a fresh sibling.
	tmp := outDir + ".tmp"
	_ = os.RemoveAll(tmp)
	if err := os.MkdirAll(tmp, 0o755); err != nil {
		return nil, err
	}
	m, err := ExportModel(srcPt, tmp, author, modelKind, notes)
	if err != nil {
		_ = os.RemoveAll(tmp)
		return nil, err
	}
	// Move children of tmp into outDir.
	entries, err := os.ReadDir(tmp)
	if err != nil {
		return nil, err
	}
	for _, e := range entries {
		from := filepath.Join(tmp, e.Name())
		to := filepath.Join(outDir, e.Name())
		if err := os.Rename(from, to); err != nil {
			return nil, fmt.Errorf("move %s: %w", e.Name(), err)
		}
	}
	_ = os.Remove(tmp)
	return m, nil
}

// RemoteBundle is one bundle found in the shared folder that hasn't yet
// been merged locally.
type RemoteBundle struct {
	MachineID string
	Timestamp string // bundle dir name (RFC3339-ish)
	Path      string // full path to the bundle dir
	Manifest  *Manifest
}

// FindNewBundles walks <sharedRoot>/<machine>/<ts>/ and returns bundles
// from machines OTHER than this one whose timestamp is newer than
// state.LastSeen[machineID]. Sorted by timestamp ascending so callers
// merging in order go oldest-first.
func FindNewBundles(sharedRoot string, state *SyncState) ([]RemoteBundle, error) {
	myID, err := MachineID()
	if err != nil {
		return nil, err
	}
	entries, err := os.ReadDir(sharedRoot)
	if err != nil {
		return nil, fmt.Errorf("read shared root: %w", err)
	}

	var out []RemoteBundle
	for _, machineEntry := range entries {
		if !machineEntry.IsDir() {
			continue
		}
		mid := machineEntry.Name()
		if mid == myID {
			continue // skip our own publishes
		}
		machineDir := filepath.Join(sharedRoot, mid)
		bundles, err := os.ReadDir(machineDir)
		if err != nil {
			continue
		}
		lastSeen := state.LastSeen[mid]
		for _, b := range bundles {
			if !b.IsDir() {
				continue
			}
			ts := b.Name()
			if lastSeen != "" && ts <= lastSeen {
				continue // already merged; lexicographic compare on RFC3339-ish ts
			}
			bundlePath := filepath.Join(machineDir, ts)
			m, err := LoadManifest(bundlePath)
			if err != nil {
				// Skip half-written or malformed bundles silently — the
				// next sync run will pick them up once they finish.
				continue
			}
			out = append(out, RemoteBundle{
				MachineID: mid, Timestamp: ts, Path: bundlePath, Manifest: m,
			})
		}
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Timestamp < out[j].Timestamp })
	return out, nil
}
