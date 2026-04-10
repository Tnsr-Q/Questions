// +build ignore

#include <linux/bpf.h>
#include <linux/pkt_cls.h>
#include <linux/ip.h>
#include <linux/tcp.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_endian.h>

/* 
 * Darwinian Routing Map
 * Key:   Virtual IP (The "Target" the Go app sends to)
 * Value: Real Alpha Pod IP (Updated by Overseer)
 */
struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, 1); // We only need 1 active Alpha at a time for this design
    __type(key, __u32);
    __type(value, __u32);
} routing_map SEC(".maps");

/*
 * Traffic Control (TC) Egress Hook
 * Attached to the eth0 interface of the Gateway Pod.
 */
SEC("tc_egress")
int route_packets(struct __sk_buff *skb) {
    void *data = (void *)(long)skb->data;
    void *data_end = (void *)(long)skb->data_end;

    struct ethhdr *eth = data;
    if ((void *)(eth + 1) > data_end) return TC_ACT_OK;

    // Only handle IP packets
    if (eth->h_proto != bpf_htons(ETH_P_IP)) return TC_ACT_OK;

    struct iphdr *ip = (void *)(eth + 1);
    if ((void *)(ip + 1) > data_end) return TC_ACT_OK;

    // Check if the destination is our Virtual IP (10.99.99.99)
    // 10.99.99.99 in Little Endian Hex is 0x6363630A
    // Ideally, this is passed via constants or map lookup, hardcoded here for clarity.
    __u32 target_vip = 0x6363630A; 

    if (ip->daddr == target_vip) {
        
        // Look up the current Alpha IP from the map
        __u32 *real_ip = bpf_map_lookup_elem(&routing_map, &target_vip);
        
        if (real_ip) {
            // DETECTED: Outgoing packet to VIP.
            // ACTION: Rewrite Destination IP to Real Alpha IP.
            
            // Recalculate IP Checksum automatically
            long ret = bpf_skb_store_bytes(skb, offsetof(struct iphdr, daddr), real_ip, sizeof(__u32), 0);
            
            if (ret < 0) return TC_ACT_SHOT; // Drop if rewrite fails

            // Note: In a real L2 environment, we might need to resolve MAC addresses here.
            // In K8s overlay networks (Flannel/Calico), IP rewrite is often sufficient 
            // before the packet hits the CNI encapsulation.
        }
    }

    return TC_ACT_OK;
}

char __license[] SEC("license") = "GPL";