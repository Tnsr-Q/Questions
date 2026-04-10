#ifndef __TENSORQ_MAPS_H__
#define __TENSORQ_MAPS_H__

#include <bpf/bpf_helpers.h>
#include "route_entry.h"

/* Stats counters exposed to userspace via the stats_map. */
struct route_counters {
	__u64 hits;     /* packets matched and rewritten */
	__u64 misses;   /* packets not found in routing_map */
	__u64 errors;   /* checksum or rewrite failures */
};

struct {
	__uint(type, BPF_MAP_TYPE_HASH);
	__uint(max_entries, 1024);
	__type(key, __u32);                  /* VIP, network byte order */
	__type(value, struct route_entry);
} routing_map SEC(".maps");

struct {
	__uint(type, BPF_MAP_TYPE_ARRAY);
	__uint(max_entries, 1);
	__type(key, __u32);
	__type(value, struct route_counters);
} stats_map SEC(".maps");

#endif /* __TENSORQ_MAPS_H__ */