# Setting Up Wails UI

## Prerequisites
- Go 1.22+
- Node.js 18+
- npm

## Install Wails CLI
```bash
go install github.com/wailsapp/wails/v2/cmd/wails@latest
```

## Initialize Frontend
```bash
cd frontend
npm install
```

## Development
```bash
wails dev
```

## Build
```bash
wails build
```

## Notes
- The App struct in `internal/app/app.go` has Wails-ready bindings
- TODO comments mark where Wails runtime calls need to be added
- The ProcessBridge (`--mock` flag not set) requires Python 3.11+ with ML dependencies
