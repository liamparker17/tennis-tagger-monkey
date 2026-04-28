// Package telemetry wires Sentry into the Go pipeline so panics from end-user
// machines get captured. Mirrors ml/_telemetry.py: same DSN resolution
// (env var TENNIS_TAGGER_SENTRY_DSN, then sentry.dsn next to the binary or
// in the repo root), same opt-out (TENNIS_TAGGER_TELEMETRY=off), same PII
// scrubbing approach.
//
// Init is a no-op when no DSN is configured, so source builds and tests
// don't need Sentry installed/configured to run.
package telemetry

import (
	"errors"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"time"

	"github.com/getsentry/sentry-go"
)

const flushTimeout = 2 * time.Second

// Init configures Sentry once. Returns true if Sentry is now active.
// `component` is logged as a tag (e.g. "tagger" for the Go pipeline).
func Init(component string) bool {
	if strings.EqualFold(os.Getenv("TENNIS_TAGGER_TELEMETRY"), "off") {
		return false
	}
	dsn := resolveDSN()
	if dsn == "" {
		return false
	}
	release := os.Getenv("TENNIS_TAGGER_VERSION")
	if release == "" {
		release = "dev"
	}
	env := os.Getenv("TENNIS_TAGGER_ENV")
	if env == "" {
		env = "production"
	}
	err := sentry.Init(sentry.ClientOptions{
		Dsn:              dsn,
		Release:          "tennis-tagger@" + release,
		Environment:      env,
		AttachStacktrace: true,
		// Crash reporting only — no perf/transaction overhead.
		EnableTracing: false,
		BeforeSend:    scrubEvent,
	})
	if err != nil {
		// Never let telemetry init kill the app.
		return false
	}
	sentry.ConfigureScope(func(scope *sentry.Scope) {
		scope.SetTag("component", component)
		scope.SetTag("go_version", runtime.Version())
		scope.SetTag("platform", runtime.GOOS+"/"+runtime.GOARCH)
	})
	return true
}

// Flush waits up to flushTimeout for queued events to be delivered.
// Call from main via defer so a panic-on-exit still gets reported.
func Flush() { sentry.Flush(flushTimeout) }

// CaptureRecover is intended to be called from a defer at the top of main:
//
//	defer telemetry.CaptureRecover()
//
// If a panic propagates this far, it'll be reported with stack trace and
// then re-thrown so the process still exits with the original failure.
func CaptureRecover() {
	if r := recover(); r != nil {
		err, ok := r.(error)
		if !ok {
			err = errors.New(asString(r))
		}
		sentry.CurrentHub().Recover(err)
		sentry.Flush(flushTimeout)
		panic(r)
	}
}

// CaptureError reports a non-fatal error.
func CaptureError(err error) {
	if err == nil {
		return
	}
	sentry.CaptureException(err)
}

func resolveDSN() string {
	if d := strings.TrimSpace(os.Getenv("TENNIS_TAGGER_SENTRY_DSN")); d != "" {
		return d
	}
	// Look for sentry.dsn next to the binary first (installed bundle), then
	// walk up to the working directory (source-tree run).
	if exe, err := os.Executable(); err == nil {
		if d := readDSNFile(filepath.Join(filepath.Dir(exe), "sentry.dsn")); d != "" {
			return d
		}
	}
	if cwd, err := os.Getwd(); err == nil {
		dir := cwd
		for i := 0; i < 6; i++ {
			if d := readDSNFile(filepath.Join(dir, "sentry.dsn")); d != "" {
				return d
			}
			parent := filepath.Dir(dir)
			if parent == dir {
				break
			}
			dir = parent
		}
	}
	return ""
}

func readDSNFile(p string) string {
	b, err := os.ReadFile(p)
	if err != nil {
		return ""
	}
	return strings.TrimSpace(string(b))
}

// scrubEvent masks the user's home directory and username in any string
// field of the outgoing event. Match-video filenames contain real player
// names; user paths contain the OS account name. Keep stack frames but
// strip identity.
func scrubEvent(event *sentry.Event, _ *sentry.EventHint) *sentry.Event {
	home, _ := os.UserHomeDir()
	user := filepath.Base(home)
	if home == "" && user == "" {
		return event
	}
	scrub := func(s string) string {
		out := s
		if home != "" {
			out = strings.ReplaceAll(out, home, "<HOME>")
		}
		if len(user) > 2 {
			out = strings.ReplaceAll(out, user, "<USER>")
		}
		return out
	}
	event.Message = scrub(event.Message)
	for i := range event.Exception {
		event.Exception[i].Value = scrub(event.Exception[i].Value)
		if event.Exception[i].Stacktrace != nil {
			for j := range event.Exception[i].Stacktrace.Frames {
				f := &event.Exception[i].Stacktrace.Frames[j]
				f.Filename = scrub(f.Filename)
				f.AbsPath = scrub(f.AbsPath)
			}
		}
	}
	return event
}

func asString(v any) string {
	if s, ok := v.(string); ok {
		return s
	}
	if e, ok := v.(error); ok {
		return e.Error()
	}
	return "panic"
}
