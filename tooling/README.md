# Tooling

This directory contains a simple CLI written in the Go Programming Language for working with barcodes and layouts.

## Installation

You need to have the Go Programming Language (v1.22 or compatible) installed on your system (https://go.dev/).

Build it using `go build -o tooling.exe main.go` from this directory.

## Example Usages

Verify, that a stream of layout entries is a valid for a layout of the given dimensions.

```bash
$ cat my_barcode_list.txt | ./tooling.exe layout verify -r 1024 -c 768
the input was a valid 1024x768 layout with unique entries of length 34
```

Convert a stream of barcodes to a stream of schedules.

```bash
$ cat my_barcode_list.txt | ./tooling.exe barcode schedule
```

You can also combine these.

```bash
$ cat my_barcode_list.txt | ./tooling.exe barcode schedule | ./tooling.exe layout verify -r 1024 -c 768
```