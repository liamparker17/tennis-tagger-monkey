package bridge

import (
	"fmt"
	"os"
	"path/filepath"
	"sync"
)

// SharedMemBuffer manages a memory-mapped file for zero-copy frame transfer
// between Go and Python. Go writes raw RGB frames to the file, Python reads
// them via numpy.memmap.
type SharedMemBuffer struct {
	dir      string
	file     *os.File
	path     string
	size     int
	mu       sync.Mutex
}

// NewSharedMemBuffer creates a shared memory buffer file in the given directory.
func NewSharedMemBuffer(dir string) (*SharedMemBuffer, error) {
	if err := os.MkdirAll(dir, 0755); err != nil {
		return nil, fmt.Errorf("create shm dir: %w", err)
	}

	path := filepath.Join(dir, "frame_buffer.raw")
	f, err := os.Create(path)
	if err != nil {
		return nil, fmt.Errorf("create shm file: %w", err)
	}

	return &SharedMemBuffer{
		dir:  dir,
		file: f,
		path: path,
	}, nil
}

// WriteBatch writes a batch of raw RGB frames to the shared buffer file.
// Returns the file path and frame metadata (offsets, dimensions) for Python.
func (s *SharedMemBuffer) WriteBatch(frames []Frame) (string, []FrameMeta, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	// Calculate total size needed
	totalSize := 0
	for _, f := range frames {
		totalSize += len(f.Data)
	}

	// Truncate and rewrite from beginning (reuse file each batch)
	if err := s.file.Truncate(0); err != nil {
		return "", nil, fmt.Errorf("truncate: %w", err)
	}
	if _, err := s.file.Seek(0, 0); err != nil {
		return "", nil, fmt.Errorf("seek: %w", err)
	}

	metas := make([]FrameMeta, len(frames))
	offset := 0
	for i, f := range frames {
		n, err := s.file.Write(f.Data)
		if err != nil {
			return "", nil, fmt.Errorf("write frame %d: %w", i, err)
		}
		metas[i] = FrameMeta{
			Offset: offset,
			Width:  f.Width,
			Height: f.Height,
			Size:   n,
		}
		offset += n
	}

	// Sync to ensure Python can read immediately
	if err := s.file.Sync(); err != nil {
		return "", nil, fmt.Errorf("sync: %w", err)
	}

	return s.path, metas, nil
}

// Path returns the shared memory file path.
func (s *SharedMemBuffer) Path() string {
	return s.path
}

// Close removes the shared memory file.
func (s *SharedMemBuffer) Close() error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.file != nil {
		s.file.Close()
		s.file = nil
	}
	os.Remove(s.path)
	return nil
}

// FrameMeta describes a single frame's location in the shared buffer.
type FrameMeta struct {
	Offset int `json:"offset"`
	Width  int `json:"width"`
	Height int `json:"height"`
	Size   int `json:"size"`
}
