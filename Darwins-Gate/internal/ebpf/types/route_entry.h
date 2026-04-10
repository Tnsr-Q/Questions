#ifndef __TENSORQ_ROUTE_ENTRY_H__
#define __TENSORQ_ROUTE_ENTRY_H__

#include <linux/types.h>

struct route_entry {
    __u32 dst_ip;            /* final L3 destination */
    __u32 next_hop_ip;       /* optional, if routed hop matters */
    __u32 ifindex;           /* egress interface */
    __u8  dst_mac[6];        /* next-hop MAC */
    __u8  src_mac[6];        /* optional source MAC override */
};

#endif /* __TENSORQ_ROUTE_ENTRY_H__ */