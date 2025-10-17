/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 27, 2023.
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
#ifndef __PBUF_H__
#define __PBUF_H__

#include <sys/mbuf.h>

enum pbuf_type {
	PBUF_TYPE_ZOMBIE = 0,
	PBUF_TYPE_MBUF,
	PBUF_TYPE_MEMORY
};

enum pbuf_action {
	PBUF_ACTION_DESTROY
};

#define PBUF_ACTION_RV_SUCCESS  0
#define PBUF_ACTION_RV_FAILURE  (-1)

struct pbuf_memory {
	uint8_t *__sized_by(pm_buffer_len) pm_buffer;     // Pointer to start of buffer
	u_int pm_buffer_len;    // Total length of buffer
	u_int pm_offset;        // Offset to start of payload
	u_int pm_len;           // Length of payload
	uint32_t pm_csum_flags;
	uint32_t pm_csum_data;
	uint8_t pm_proto;
	uint8_t pm_flowsrc;
	uint32_t pm_flowid;
	uint32_t pm_flow_gencnt;
	uint32_t pm_flags;
	struct pf_mtag pm_pftag;
	struct pf_fragment_tag  pm_pf_fragtag;
	int (*pm_action)(struct pbuf_memory *, enum pbuf_action);
	void *pm_action_cookie;
};

typedef struct pbuf {
	enum pbuf_type  pb_type;
	union {
		struct mbuf *pbu_mbuf;
		struct pbuf_memory pbu_memory;
	} pb_u;
#define pb_mbuf         pb_u.pbu_mbuf
#define pb_memory       pb_u.pbu_memory

	void            *__sized_by(pb_contig_len) pb_data;
	uint32_t        pb_packet_len;
	uint32_t        pb_contig_len;
	uint32_t        *pb_csum_flags;
	uint32_t        *pb_csum_data;    /* data field used by csum routines */
	uint8_t         *pb_proto;
	uint8_t         *pb_flowsrc;
	uint32_t        *pb_flowid;
	uint32_t        *pb_flow_gencnt;
	uint32_t        *pb_flags;
	struct pf_mtag  *pb_pftag;
	struct pf_fragment_tag  *pb_pf_fragtag;
	struct ifnet    *pb_ifp;
	struct pbuf     *pb_next;
} pbuf_t;

#define pbuf_is_valid(pb) (!((pb) == NULL || (pb)->pb_type == PBUF_TYPE_ZOMBIE))

void            pbuf_init_mbuf(pbuf_t *, struct mbuf *, struct ifnet *);
void            pbuf_init_memory(pbuf_t *, const struct pbuf_memory *,
    struct ifnet *);
void            pbuf_destroy(pbuf_t *);
void            pbuf_sync(pbuf_t *);

struct mbuf     *pbuf_to_mbuf(pbuf_t *, boolean_t);
struct mbuf     *pbuf_clone_to_mbuf(pbuf_t *);

void *          pbuf_ensure_contig(pbuf_t *, size_t);
void *          pbuf_ensure_writable(pbuf_t *, size_t);

void *          pbuf_resize_segment(pbuf_t *, int off, int olen, int nlen);
void *          pbuf_contig_segment(pbuf_t *, int off, int len);

void            pbuf_copy_data(pbuf_t *, int, int, void *__sized_by(buflen), size_t buflen);
void            pbuf_copy_back(pbuf_t *, int, int, void *__sized_by(buflen), size_t buflen);

uint16_t        pbuf_inet_cksum(const pbuf_t *, uint32_t, uint32_t, uint32_t);
uint16_t        pbuf_inet6_cksum(const pbuf_t *, uint32_t, uint32_t, uint32_t);

mbuf_svc_class_t pbuf_get_service_class(const pbuf_t *);
void *          pbuf_get_packet_buffer_address(const pbuf_t *);
#endif /* __PBUF_H__ */
