.PHONY: test clean build run

test:
	go test ./internal/... -v

clean:
	rm -rf build/
	go clean

build:
	go build -o build/tagger ./cmd/tagger

run:
	go run ./cmd/tagger
