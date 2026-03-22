package video

// Frame represents a single video frame with raw RGB pixel data.
type Frame struct {
	Data   []byte // Raw RGB bytes, length = Width * Height * 3
	Width  int
	Height int
}

// VideoMeta holds metadata about a video file.
type VideoMeta struct {
	Path         string
	FPS          float64
	Width        int // Scaled output width (640px for detection)
	Height       int // Scaled output height
	NativeWidth  int // Original video width
	NativeHeight int // Original video height
	TotalFrames  int
	Duration     float64 // seconds
}
