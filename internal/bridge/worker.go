package bridge

import (
	"encoding/json"
	"runtime"
	"sync"
)

// bridgeResponse carries the result of a bridge call back to the caller.
type bridgeResponse struct {
	data json.RawMessage
	err  error
}

// bridgeRequest represents a single method call to be dispatched on the worker goroutine.
type bridgeRequest struct {
	method   string
	payload  json.RawMessage
	response chan bridgeResponse
}

// pythonCaller is the low-level interface for calling into a Python backend.
// Implementations include StubCaller, embeddedCaller, and processCaller.
type pythonCaller interface {
	call(method string, payload json.RawMessage) (json.RawMessage, error)
	init(config BridgeConfig) error
	close()
}

// Worker serializes all bridge calls onto a single OS thread, which is required
// for Python's GIL when using embedded CPython, and simplifies subprocess I/O.
type Worker struct {
	requests chan bridgeRequest
	done     chan struct{}
	backend  pythonCaller
	stopOnce sync.Once
}

// NewWorker creates and starts a Worker that dispatches calls to the given backend.
// The worker goroutine is pinned to a single OS thread via runtime.LockOSThread().
func NewWorker(backend pythonCaller) *Worker {
	w := &Worker{
		requests: make(chan bridgeRequest),
		done:     make(chan struct{}),
		backend:  backend,
	}
	go w.run()
	return w
}

// run is the worker loop. It locks the current goroutine to an OS thread and
// processes requests sequentially until the requests channel is closed.
func (w *Worker) run() {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	defer close(w.done)

	for req := range w.requests {
		data, err := w.backend.call(req.method, req.payload)
		req.response <- bridgeResponse{data: data, err: err}
	}
}

// Call sends a method invocation to the worker and blocks until the response is ready.
func (w *Worker) Call(method string, payload json.RawMessage) (json.RawMessage, error) {
	resp := make(chan bridgeResponse, 1)
	w.requests <- bridgeRequest{
		method:   method,
		payload:  payload,
		response: resp,
	}
	r := <-resp
	return r.data, r.err
}

// Stop closes the requests channel and waits for the worker goroutine to finish.
// Safe to call multiple times.
func (w *Worker) Stop() {
	w.stopOnce.Do(func() {
		close(w.requests)
		<-w.done
	})
}

// StubCaller is a pythonCaller that returns an empty JSON object for all calls.
// Useful for tests and as a placeholder before real Python integration.
type StubCaller struct{}

func (s *StubCaller) call(method string, payload json.RawMessage) (json.RawMessage, error) {
	return json.RawMessage(`{}`), nil
}

func (s *StubCaller) init(config BridgeConfig) error {
	return nil
}

func (s *StubCaller) close() {}
