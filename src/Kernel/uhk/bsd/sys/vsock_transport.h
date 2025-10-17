/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 5, 2023.
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
#ifndef _VSOCK_TRANSPORT_H_
#define _VSOCK_TRANSPORT_H_
#ifdef  KERNEL_PRIVATE

#include <sys/cdefs.h>

__BEGIN_DECLS

#include <sys/queue.h>
#include <sys/kernel_types.h>
#include <sys/vsock.h>

#define VSOCK_MAX_PACKET_SIZE 65536

enum vsock_operation {
	VSOCK_REQUEST = 0,
	VSOCK_RESPONSE = 1,
	VSOCK_PAYLOAD = 2,
	VSOCK_SHUTDOWN = 3,
	VSOCK_SHUTDOWN_RECEIVE = 4,
	VSOCK_SHUTDOWN_SEND = 5,
	VSOCK_RESET = 6,
	VSOCK_CREDIT_UPDATE = 7,
	VSOCK_CREDIT_REQUEST = 8,
};

struct vsock_address {
	uint32_t cid;
	uint32_t port;
};

struct vsock_transport {
	void *provider;
	int (*get_cid)(void *provider, uint32_t *cid);
	int (*attach_socket)(void *provider);
	int (*detach_socket)(void *provider);
	int (*put_message)(void *provider, struct vsock_address src, struct vsock_address dst,
	    enum vsock_operation op, uint32_t buf_alloc, uint32_t fwd_cnt, mbuf_t m);
};

extern int vsock_add_transport(struct vsock_transport *transport);
extern int vsock_remove_transport(struct vsock_transport *transport);
extern int vsock_reset_transport(struct vsock_transport *transport);
extern int vsock_put_message(struct vsock_address src, struct vsock_address dst,
    enum vsock_operation op, uint32_t buf_alloc, uint32_t fwd_cnt, mbuf_t m);

__END_DECLS

#endif /* KERNEL_PRIVATE */
#endif /* _VSOCK_TRANSPORT_H_ */
