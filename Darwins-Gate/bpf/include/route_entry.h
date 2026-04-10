/*
 * TensorQ Darwinian Gateway — Route Entry Definition
 *
 * This header is shared between:
 *   - eBPF C code (router_tc.c) via #include "../include/route_entry.h"
 *   - Go code via byte-for-byte struct matching in types/types.go
 *
 * Layout: 24 bytes, NO padding.
 * All fields are explicitly packed.
 */
#ifndef __TENSORQ_ROUTE_ENTRY_H__
#define __TENSORQ_ROUTE_ENTRY_H__

#include <linux/types.h>

/*
 * struct route_entry — VIP-to-backend routing entry stored in eBPF HASH map.
 *
 * @dst_ip:         New destination IPv4 (big-endian) — the pod IP to rewrite to.
 * @next_hop_ip:    Optional next-hop IPv4 (big-endian). Currently set to dst_ip.
 * @ifindex:        Egress interface index. 0 = use kernel default.
 * @dst_mac:        Next-hop destination MAC (6 bytes). Used to rewrite eth header.
 * @src_mac:        Optional source MAC override (6 bytes). Zero = no change.
 *
 * Total: 4 + 4 + 4 + 6 + 6 = 24 bytes.
 */
struct route_entry {
	__u32 dst_ip;            /* offset 0 — new L3 destination */
	__u32 next_hop_ip;       /* offset 4 — next-hop IP (or 0) */
	__u32 ifindex;           /* offset 8 — egress interface index */
	__u8  dst_mac[6];        /* offset 12 — next-hop MAC */
	__u8  src_mac[6];        /* offset 18 — source MAC override */
};

#endif /* __TENSORQ_ROUTE_ENTRY_H__ */
