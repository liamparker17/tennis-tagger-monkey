// pmprobe is a throwaway smoke driver for the Plan 4 Go↔Python inference loop.
// Usage: pmprobe <python> <ckpt> <clip.mp4>
package main

import (
	"encoding/json"
	"fmt"
	"os"

	"github.com/liamp/tennis-tagger/internal/pointmodel"
)

func main() {
	if len(os.Args) != 4 {
		fmt.Fprintln(os.Stderr, "usage: pmprobe <python> <ckpt> <clip.mp4>")
		os.Exit(2)
	}
	c, err := pointmodel.Start(os.Args[1], os.Args[2])
	if err != nil {
		fmt.Fprintln(os.Stderr, "start:", err)
		os.Exit(1)
	}
	defer c.Close()
	if err := c.Ping(); err != nil {
		fmt.Fprintln(os.Stderr, "ping:", err)
		os.Exit(1)
	}
	pred, err := c.PredictPoint(os.Args[3])
	if err != nil {
		fmt.Fprintln(os.Stderr, "predict:", err)
		os.Exit(1)
	}
	b, _ := json.MarshalIndent(pred, "", "  ")
	fmt.Println(string(b))
}
