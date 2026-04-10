package types

import (
	"encoding/binary"
	"fmt"
	"net"
)

// RouteEntry MUST match the C struct in route_entry.h byte-for-byte.
// C layout (24 bytes, no padding):
//
//	struct route_entry {
//	    __u32 dst_ip;            // offset 0
//	    __u32 next_hop_ip;       // offset 4
//	    __u32 ifindex;           // offset 8
//	    __u8  dst_mac[6];        // offset 12
//	    __u8  src_mac[6];        // offset 18
//	};
//
// Go guarantees: no padding between uint32 fields, [6]byte is packed,
// total = 4+4+4+6+6 = 24 bytes.
type RouteEntry struct {
	DstIP     uint32   // offset 0 — big-endian IPv4
	NextHopIP uint32   // offset 4 — next-hop IPv4 (or 0)
	Ifindex   uint32   // offset 8 — egress interface index
	DstMAC    [6]byte  // offset 12 — next-hop MAC
	SrcMAC    [6]byte  // offset 18 — source MAC override (or zeros)
}

type RouteCounters struct {
	Hits   uint64
	Misses uint64
	Errors uint64
}

func IPv4ToBE32(ipStr string) (uint32, error) {
	ip := net.ParseIP(ipStr)
	if ip == nil {
		return 0, fmt.Errorf("invalid IP: %q", ipStr)
	}
	ip4 := ip.To4()
	if ip4 == nil {
		return 0, fmt.Errorf("not an IPv4 address: %q", ipStr)
	}
	return binary.BigEndian.Uint32(ip4), nil
}

func MustMAC(macStr string) ([6]byte, error) {
	var out [6]byte
	hw, err := net.ParseMAC(macStr)
	if err != nil {
		return out, fmt.Errorf("invalid MAC %q: %w", macStr, err)
	}
	if len(hw) != 6 {
		return out, fmt.Errorf("expected 6-byte MAC, got %d bytes", len(hw))
	}
	copy(out[:], hw)
	return out, nil
}

func NewRouteEntry(dstIP string, nextHopIP string, ifindex uint32, dstMAC string, srcMAC string) (RouteEntry, error) {
	var e RouteEntry

	ip, err := IPv4ToBE32(dstIP)
	if err != nil {
		return e, err
	}
	mac, err := MustMAC(dstMAC)
	if err != nil {
		return e, err
	}

	e.DstIP = ip

	if nextHopIP != "" {
		nh, err := IPv4ToBE32(nextHopIP)
		if err != nil {
			return e, err
		}
		e.NextHopIP = nh
	}

	e.Ifindex = ifindex
	e.DstMAC = mac

	if srcMAC != "" {
		sm, err := MustMAC(srcMAC)
		if err != nil {
			return e, err
		}
		e.SrcMAC = sm
	}

	return e, nil
}