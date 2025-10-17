/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 23, 2025.
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
/* Copyright (c) 1997, 1998 Apple Computer, Inc. All Rights Reserved */
/*
 *	@(#)ndrv.h	1.1 (MacOSX) 6/10/43
 * Justin Walker - 970604
 */

#ifndef _NET_NDRV_VAR_H
#define _NET_NDRV_VAR_H
#ifdef PRIVATE

#if BSD_KERNEL_PRIVATE
/*
 * structure for storing a linked list of multicast addresses
 * registered by this socket. May be variable in length.
 */
struct ndrv_multiaddr {
	struct ndrv_multiaddr      *next;
	ifmultiaddr_t               ifma;
	struct sockaddr            *addr;
};
#endif

/*
 * The cb is plugged into the socket (so_pcb), and the ifnet structure
 *  of BIND is plugged in here.
 * For now, it looks like a raw_cb up front...
 */
struct ndrv_cb {
	TAILQ_ENTRY(ndrv_cb)    nd_next;
	struct socket *nd_socket;       /* Back to the socket */
	u_int32_t nd_signature; /* Just double-checking */
	struct sockaddr_ndrv *nd_faddr;
	struct sockaddr_ndrv *nd_laddr;
	struct sockproto nd_proto;      /* proto family, protocol */
	int nd_descrcnt;                /* # elements in nd_dlist - Obsolete */
	TAILQ_HEAD(dlist, dlil_demux_desc) nd_dlist; /* Descr. list */
	u_int32_t nd_dlist_cnt; /* Descr. list count */
	struct ifnet *nd_if; /* obsolete, maintained for binary compatibility */
	u_int32_t nd_proto_family;
	u_int32_t nd_family;
	struct ndrv_multiaddr* nd_multiaddrs;
	short nd_unit;
};

#define sotondrvcb(so)          ((struct ndrv_cb *)(so)->so_pcb)
#define NDRV_SIGNATURE  0x4e445256 /* "NDRV" */

/* Nominal allocated space for NDRV sockets */
#define NDRVSNDQ         8192
#define NDRVRCVQ         8192

#endif /* PRIVATE */
#endif  /* _NET_NDRV_VAR_H */
