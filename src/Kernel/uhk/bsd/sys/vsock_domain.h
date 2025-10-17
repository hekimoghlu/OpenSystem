/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 1, 2024.
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
#ifndef _VSOCK_DOMAIN_H_
 #define _VSOCK_DOMAIN_H_
 #ifdef  BSD_KERNEL_PRIVATE

 #include <sys/queue.h>
 #include <sys/vsock_transport.h>

/* VSock Protocol Control Block */

struct vsockpcb {
	TAILQ_ENTRY(vsockpcb) all;
	LIST_ENTRY(vsockpcb) bound;
	struct socket *so;
	struct vsock_address local_address;
	struct vsock_address remote_address;
	struct vsock_transport *transport;
	uint32_t fwd_cnt;
	uint32_t tx_cnt;
	uint32_t peer_buf_alloc;
	uint32_t peer_fwd_cnt;
	uint32_t last_buf_alloc;
	uint32_t last_fwd_cnt;
	size_t waiting_send_size;
	vsock_gen_t vsock_gencnt;
};

/* VSock Protocol Control Block Info */

struct vsockpcbinfo {
	// PCB locking.
	lck_rw_t all_lock;
	lck_rw_t bound_lock;
	// PCB lists.
	TAILQ_HEAD(, vsockpcb) all;
	LIST_HEAD(, vsockpcb) bound;
	// Port generation.
	uint32_t last_port;
	lck_mtx_t port_lock;
	// Counts.
	uint64_t all_pcb_count;
	vsock_gen_t vsock_gencnt;
};

#endif /* BSD_KERNEL_PRIVATE */
#endif /* _VSOCK_DOMAIN_H_ */
