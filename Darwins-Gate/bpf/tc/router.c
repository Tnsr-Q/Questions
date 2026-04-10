#include <linux/bpf.h>
#include <linux/if_ether.h>
#include <linux/ip.h>
#include <linux/in.h>
#include <linux/pkt_cls.h>
#include <linux/udp.h>
#include <linux/tcp.h>
#include <linux/if_vlan.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_endian.h>
#include "include/route_entry.h"
#include "include/maps.h"

struct route_entry {
    __u32 dst_ip;               /* network byte order */
    unsigned char dst_mac[6];   /* next-hop MAC, not abstract node MAC */
    __u16 pad;
};

struct counters {
    __u64 hits;
    __u64 misses;
    __u64 errors;
};

struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, 1024);
    __type(key, __u32);
    __type(value, struct route_entry);
} routing_map SEC(".maps");

struct {
    __uint(type, BPF_MAP_TYPE_ARRAY);
    __uint(max_entries, 1);
    __type(key, __u32);
    __type(value, struct counters);
} stats_map SEC(".maps");

static __always_inline void bump_hit(void) {
    __u32 k = 0;
    struct counters *c = bpf_map_lookup_elem(&stats_map, &k);
    if (c)
        __sync_fetch_and_add(&c->hits, 1);
}

static __always_inline void bump_miss(void) {
    __u32 k = 0;
    struct counters *c = bpf_map_lookup_elem(&stats_map, &k);
    if (c)
        __sync_fetch_and_add(&c->misses, 1);
}

static __always_inline void bump_error(void) {
    __u32 k = 0;
    struct counters *c = bpf_map_lookup_elem(&stats_map, &k);
    if (c)
        __sync_fetch_and_add(&c->errors, 1);
}

static __always_inline int parse_eth(void *data, void *data_end, __u16 *proto, __u64 *l3_off) {
    struct ethhdr *eth = data;

    if ((void *)(eth + 1) > data_end)
        return -1;

    *proto = bpf_ntohs(eth->h_proto);
    *l3_off = sizeof(*eth);

    if (*proto == ETH_P_8021Q || *proto == ETH_P_8021AD) {
        struct vlan_hdr *vh = data + *l3_off;
        if ((void *)(vh + 1) > data_end)
            return -1;
        *proto = bpf_ntohs(vh->h_vlan_encapsulated_proto);
        *l3_off += sizeof(*vh);
    }

    return 0;
}

SEC("tc")
int route_packets(struct __sk_buff *skb) {
    void *data = (void *)(long)skb->data;
    void *data_end = (void *)(long)skb->data_end;

    __u16 proto = 0;
    __u64 l3_off = 0;

    if (parse_eth(data, data_end, &proto, &l3_off) < 0)
        return TC_ACT_OK;

    if (proto != ETH_P_IP)
        return TC_ACT_OK;

    struct iphdr *ip = data + l3_off;
    if ((void *)(ip + 1) > data_end)
        return TC_ACT_OK;

    if (ip->version != 4)
        return TC_ACT_OK;

    __u32 ihl_len = ip->ihl * 4;
    if (ihl_len < sizeof(*ip))
        return TC_ACT_OK;
    if ((void *)ip + ihl_len > data_end)
        return TC_ACT_OK;

    __u32 vip = ip->daddr;
    struct route_entry *target = bpf_map_lookup_elem(&routing_map, &vip);
    if (!target) {
        bump_miss();
        return TC_ACT_OK;
    }

    /* Save old/new IPs before helper calls */
    __be32 old_daddr = ip->daddr;
    __be32 new_daddr = target->dst_ip;

    /* L3 checksum update */
    long l3_diff = bpf_csum_diff(&old_daddr, sizeof(old_daddr),
                                 &new_daddr, sizeof(new_daddr), 0);

    if (bpf_l3_csum_replace(skb,
            l3_off + offsetof(struct iphdr, check),
            0, l3_diff, 0) < 0) {
        bump_error();
        return TC_ACT_SHOT;
    }

    /* L4 checksum update if TCP/UDP and checksum is present */
    if (ip->protocol == IPPROTO_TCP) {
        __u64 l4_off = l3_off + ihl_len;
        struct tcphdr *tcp = data + l4_off;
        if ((void *)(tcp + 1) <= data_end) {
            if (bpf_l4_csum_replace(skb,
                    l4_off + offsetof(struct tcphdr, check),
                    0, l3_diff, BPF_F_PSEUDO_HDR | sizeof(new_daddr)) < 0) {
                bump_error();
                return TC_ACT_SHOT;
            }
        }
    } else if (ip->protocol == IPPROTO_UDP) {
        __u64 l4_off = l3_off + ihl_len;
        struct udphdr *udp = data + l4_off;
        if ((void *)(udp + 1) <= data_end && udp->check) {
            if (bpf_l4_csum_replace(skb,
                    l4_off + offsetof(struct udphdr, check),
                    0, l3_diff, BPF_F_PSEUDO_HDR | sizeof(new_daddr)) < 0) {
                bump_error();
                return TC_ACT_SHOT;
            }
        }
    }

    if (bpf_skb_store_bytes(skb,
            l3_off + offsetof(struct iphdr, daddr),
            &new_daddr, sizeof(new_daddr), 0) < 0) {
        bump_error();
        return TC_ACT_SHOT;
    }

    if (bpf_skb_store_bytes(skb,
            offsetof(struct ethhdr, h_dest),
            target->dst_mac, ETH_ALEN, 0) < 0) {
        bump_error();
        return TC_ACT_SHOT;
    }

    bump_hit();
    return TC_ACT_OK;
}

char __license[] SEC("license") = "GPL";