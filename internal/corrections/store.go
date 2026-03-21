package corrections

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"
)

// Store persists corrections as individual JSON files in a directory.
type Store struct {
	dir       string
	threshold int
}

// NewStore creates a new correction store. It ensures the storage directory
// exists and sets the default retrain threshold to 100.
func NewStore(dir string) *Store {
	_ = os.MkdirAll(dir, 0o755)
	return &Store{
		dir:       dir,
		threshold: 100,
	}
}

// SetThreshold overrides the default retrain threshold.
func (s *Store) SetThreshold(n int) {
	s.threshold = n
}

// Save writes a correction to disk as a JSON file. It auto-generates an ID
// (UnixNano) and Timestamp if they are empty.
func (s *Store) Save(c Correction) error {
	if c.ID == "" {
		c.ID = strconv.FormatInt(time.Now().UnixNano(), 10)
	}
	if c.Timestamp == "" {
		c.Timestamp = time.Now().UTC().Format(time.RFC3339)
	}

	data, err := json.MarshalIndent(c, "", "  ")
	if err != nil {
		return fmt.Errorf("marshal correction: %w", err)
	}

	path := filepath.Join(s.dir, c.ID+".json")
	if err := os.WriteFile(path, data, 0o644); err != nil {
		return fmt.Errorf("write correction file: %w", err)
	}
	return nil
}

// List reads all correction JSON files and returns them sorted newest first
// (by filename / ID which is a UnixNano timestamp).
func (s *Store) List() ([]Correction, error) {
	entries, err := os.ReadDir(s.dir)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, fmt.Errorf("read corrections dir: %w", err)
	}

	var corrections []Correction
	for _, e := range entries {
		if e.IsDir() || !strings.HasSuffix(e.Name(), ".json") {
			continue
		}
		data, err := os.ReadFile(filepath.Join(s.dir, e.Name()))
		if err != nil {
			return nil, fmt.Errorf("read correction %s: %w", e.Name(), err)
		}
		var c Correction
		if err := json.Unmarshal(data, &c); err != nil {
			return nil, fmt.Errorf("unmarshal correction %s: %w", e.Name(), err)
		}
		corrections = append(corrections, c)
	}

	// Sort newest first (highest ID = most recent UnixNano).
	sort.Slice(corrections, func(i, j int) bool {
		return corrections[i].ID > corrections[j].ID
	})

	return corrections, nil
}

// Count returns the number of correction JSON files in the store directory.
func (s *Store) Count() int {
	entries, err := os.ReadDir(s.dir)
	if err != nil {
		return 0
	}
	n := 0
	for _, e := range entries {
		if !e.IsDir() && strings.HasSuffix(e.Name(), ".json") {
			n++
		}
	}
	return n
}

// ShouldRetrain reports whether the number of stored corrections has reached
// the retrain threshold.
func (s *Store) ShouldRetrain() bool {
	return s.Count() >= s.threshold
}

// Flush collects all corrections, moves their files into an archived/
// subdirectory, and returns the batch. After a successful flush the store
// directory contains no correction files and Count() returns 0.
func (s *Store) Flush() (*CorrectionBatch, error) {
	corrections, err := s.List()
	if err != nil {
		return nil, err
	}
	if len(corrections) == 0 {
		return &CorrectionBatch{}, nil
	}

	archiveDir := filepath.Join(s.dir, "archived")
	if err := os.MkdirAll(archiveDir, 0o755); err != nil {
		return nil, fmt.Errorf("create archive dir: %w", err)
	}

	for _, c := range corrections {
		src := filepath.Join(s.dir, c.ID+".json")
		dst := filepath.Join(archiveDir, c.ID+".json")
		if err := os.Rename(src, dst); err != nil {
			return nil, fmt.Errorf("archive correction %s: %w", c.ID, err)
		}
	}

	return &CorrectionBatch{
		Corrections: corrections,
	}, nil
}
