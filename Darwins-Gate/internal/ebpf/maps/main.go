package main

import (
	"flag"
	"fmt"
	"log"
	"os"

	"github.com/tnsr-q/QFT-Engine/internal/ebpf/maps"
)

func main() {
	var (
		routingPath = flag.String("routing-map", maps.DefaultRoutingPath, "pinned routing map path")
		statsPath   = flag.String("stats-map", maps.DefaultStatsPath, "pinned stats map path")
		op          = flag.String("op", "", "operation: put|delete")
		vip         = flag.String("vip", "", "virtual IP")
		dstIP       = flag.String("dst-ip", "", "destination pod/backend IP")
		dstMAC      = flag.String("dst-mac", "", "next-hop MAC")
	)
	flag.Parse()

	if *op == "" || *vip == "" {
		log.Fatalf("required: -op and -vip")
	}

	rm, err := maps.OpenPinned(*routingPath, *statsPath)
	if err != nil {
		log.Fatalf("open pinned maps: %v", err)
	}
	defer rm.Close()

	switch *op {
	case "put":
		if *dstIP == "" || *dstMAC == "" {
			log.Fatalf("put requires -dst-ip and -dst-mac")
		}
		if err := maps.UpdateRoute(rm.Routing, *vip, *dstIP, *dstMAC); err != nil {
			log.Fatalf("update route: %v", err)
		}
		fmt.Fprintf(os.Stdout, "updated vip=%s -> dst=%s mac=%s\n", *vip, *dstIP, *dstMAC)

	case "delete":
		if err := maps.DeleteRoute(rm.Routing, *vip); err != nil {
			log.Fatalf("delete route: %v", err)
		}
		fmt.Fprintf(os.Stdout, "deleted vip=%s\n", *vip)

	default:
		log.Fatalf("unknown op %q", *op)
	}
}