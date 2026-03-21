package video

// Frame represents a single video frame with raw RGB pixel data.
type Frame struct {
	Data   []byte // Raw RGB bytes, length = Width * Height * 3
	Width  int
	Height int
}

// VideoMeta holds metadata about a video file.
type VideoMeta struct {
	Path        string
	FPS         float64
	Width       int
	Height      int
	TotalFrames int
	Duration    float64 // seconds
}
