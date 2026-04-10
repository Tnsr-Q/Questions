package main

import (
	"flag"
	"log"

	"github.com/tnsr-q/QFT-Engine/internal/ebpf/runtime"
)

func main() {
	var (
		iface   = flag.String("iface", "", "network interface to attach to")
		dir     = flag.String("dir", "egress", "attach direction: ingress|egress")
		pinRoot = flag.String("pin-root", runtime.DefaultPinRoot, "bpffs pin root")
	)
	flag.Parse()

	cfg := runtime.Config{
		Interface: *iface,
		Direction: runtime.AttachDirection(*dir),
		PinRoot:   *pinRoot,
	}

	rt, err := runtime.New(cfg)
	if err != nil {
		log.Fatalf("runtime init: %v", err)
	}
	defer func() {
		if err := rt.Close(); err != nil {
			log.Printf("runtime close warning: %v", err)
		}
	}()

	if err := rt.Start(); err != nil {
		log.Fatalf("runtime start: %v", err)
	}

	log.Printf("router tc loaded on iface=%s dir=%s pin-root=%s", cfg.Interface, cfg.Direction, cfg.PinRoot)

	if err := rt.Wait(); err != nil {
		log.Fatalf("runtime wait: %v", err)
	}
} 