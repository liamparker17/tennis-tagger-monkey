package pointmodel

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"os/exec"
	"sync"
	"sync/atomic"
)

type Client struct {
	cmd    *exec.Cmd
	stdin  io.WriteCloser
	stdout *bufio.Reader
	mu     sync.Mutex
	nextID int64
}

type rpcReq struct {
	JSONRPC string         `json:"jsonrpc"`
	ID      int64          `json:"id"`
	Method  string         `json:"method"`
	Params  map[string]any `json:"params,omitempty"`
}

type rpcResp struct {
	JSONRPC string          `json:"jsonrpc"`
	ID      int64           `json:"id"`
	Result  json.RawMessage `json:"result,omitempty"`
	Error   *struct {
		Code    int    `json:"code"`
		Message string `json:"message"`
	} `json:"error,omitempty"`
}

// Start launches a Python inference server subprocess and returns a client bound to its stdio.
func Start(python string, ckpt string) (*Client, error) {
	cmd := exec.Command(python, "-m", "ml.inference_server.server", "--ckpt", ckpt)
	stdin, err := cmd.StdinPipe()
	if err != nil {
		return nil, err
	}
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return nil, err
	}
	if err := cmd.Start(); err != nil {
		return nil, err
	}
	return &Client{cmd: cmd, stdin: stdin, stdout: bufio.NewReader(stdout)}, nil
}

// StartCmd wires a client to an already-constructed *exec.Cmd, useful for tests
// that want to swap in a fake subprocess.
func StartCmd(cmd *exec.Cmd) (*Client, error) {
	stdin, err := cmd.StdinPipe()
	if err != nil {
		return nil, err
	}
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return nil, err
	}
	if err := cmd.Start(); err != nil {
		return nil, err
	}
	return &Client{cmd: cmd, stdin: stdin, stdout: bufio.NewReader(stdout)}, nil
}

func (c *Client) call(method string, params map[string]any, out any) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	id := atomic.AddInt64(&c.nextID, 1)
	req := rpcReq{JSONRPC: "2.0", ID: id, Method: method, Params: params}
	b, err := json.Marshal(req)
	if err != nil {
		return err
	}
	if _, err := c.stdin.Write(append(b, '\n')); err != nil {
		return err
	}
	line, err := c.stdout.ReadBytes('\n')
	if err != nil {
		return err
	}
	var resp rpcResp
	if err := json.Unmarshal(line, &resp); err != nil {
		return err
	}
	if resp.Error != nil {
		return fmt.Errorf("rpc: %s", resp.Error.Message)
	}
	if out == nil {
		return nil
	}
	return json.Unmarshal(resp.Result, out)
}

func (c *Client) Ping() error {
	var s string
	return c.call("ping", nil, &s)
}

func (c *Client) PredictPoint(clipPath string) (*FusedPointPrediction, error) {
	var out FusedPointPrediction
	if err := c.call("predict_point", map[string]any{"clip_path": clipPath}, &out); err != nil {
		return nil, err
	}
	return &out, nil
}

func (c *Client) Close() error {
	_ = c.call("shutdown", nil, nil)
	_ = c.stdin.Close()
	if c.cmd != nil {
		return c.cmd.Wait()
	}
	return nil
}
