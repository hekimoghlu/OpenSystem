/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 2, 2022.
 *
 * Licensed under the Apache License, Version 2.0 (the ""License"");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at:
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an ""AS IS"" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Please contact NeXTHub Corporation, 651 N Broad St, Suite 201, 
 * Middletown, DE 19709, New Castle County, USA.
 *
 */
#ifndef __FLOW_DIVERT_H__
#define __FLOW_DIVERT_H__

#include <sys/mbuf.h>

#define FLOW_DIVERT_ORDER_LAST      INT_MAX

struct flow_divert_group;

struct flow_divert_trie_node {
	uint16_t                        start;
	uint16_t                        length;
	uint16_t                        child_map;
};


struct flow_divert_pcb {
	decl_lck_mtx_data(, mtx);
	socket_t                        so;
	RB_ENTRY(flow_divert_pcb)       rb_link;
	uint32_t                        hash;
	mbuf_t                          connect_token;
	uint32_t                        flags;
	uint32_t                        send_window;
	struct flow_divert_group        *group;
	uint32_t                        control_group_unit;
	uint32_t                        aggregate_unit;
	uint32_t                        policy_control_unit;
	int32_t                         ref_count;
	uint64_t                        bytes_written_by_app;
	uint64_t                        bytes_sent;
	uint64_t                        bytes_received;
	uint8_t                         log_level;
	SLIST_ENTRY(flow_divert_pcb)    tmp_list_entry;
	mbuf_t                          connect_packet;
	uint8_t                         *app_data __counted_by(app_data_length);
	size_t                          app_data_length;
	union sockaddr_in_4_6           local_endpoint;
	struct sockaddr                 *original_remote_endpoint;
	struct ifnet                    *original_last_outifp6;
	struct ifnet                    *original_last_outifp;
	uint8_t                         original_vflag;
	bool                            plugin_locked; // tell detach() that our locking is correct
};

RB_HEAD(fd_pcb_tree, flow_divert_pcb);

struct flow_divert_trie {
	struct flow_divert_trie_node    *nodes      __counted_by(nodes_count);
	uint16_t                        *child_maps __sized_by(child_maps_size);
	uint8_t                         *bytes      __counted_by(bytes_count);
	void                            *memory     __sized_by(memory_size);
	uint16_t                        nodes_count;
	uint16_t                        child_maps_count;
	uint16_t                        bytes_count;
	uint16_t                        nodes_free_next;
	uint16_t                        child_maps_free_next;
	uint16_t                        bytes_free_next;
	uint16_t                        root;
	size_t                          memory_size;
	size_t                          child_maps_size;
};

struct flow_divert_group {
	decl_lck_rw_data(, lck);
	TAILQ_ENTRY(flow_divert_group)  chain;
	struct fd_pcb_tree              pcb_tree;
	uint32_t                        ctl_unit;
	uint8_t                         atomic_bits;
	MBUFQ_HEAD(send_queue_head)     send_queue;
	uint8_t                         *token_key __counted_by(token_key_size);
	size_t                          token_key_size;
	uint32_t                        flags;
	struct flow_divert_trie         signing_id_trie;
	int32_t                         ref_count;
	pid_t                           in_process_pid;
	int32_t                         order;
};

void            flow_divert_init(void);
void            flow_divert_detach(struct socket *so);
errno_t         flow_divert_token_set(struct socket *so, struct sockopt *sopt);
errno_t         flow_divert_token_get(struct socket *so, struct sockopt *sopt);
errno_t         flow_divert_pcb_init(struct socket *so);
errno_t         flow_divert_connect_out(struct socket *so, struct sockaddr *to, proc_t p);
errno_t         flow_divert_implicit_data_out(struct socket *so, int flags, mbuf_t data, struct sockaddr *to, mbuf_t control, struct proc *p);

#endif /* __FLOW_DIVERT_H__ */
