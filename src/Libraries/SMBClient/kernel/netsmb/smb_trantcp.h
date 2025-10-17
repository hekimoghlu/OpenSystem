/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 16, 2024.
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
#ifndef _NETSMB_SMB_TRANTCP_H_
#define	_NETSMB_SMB_TRANTCP_H_

#include "netbios.h"

#ifdef _KERNEL

#ifdef NB_DEBUG
#define NBDEBUG(format, args...)	 printf("%s(%d): "format,	\
					    __FUNCTION__ , __LINE__ ,## args)
#else
#define NBDEBUG(format, args...)
#endif

enum nbstate {
	NBST_CLOSED,
	NBST_RQSENT,
	NBST_SESSION,
	NBST_RETARGET,
	NBST_REFUSED
};


#define	NBF_LOCADDR		0x0001		/* has local addr */
#define	NBF_CONNECTED	0x0002
#define	NBF_RECVLOCK	0x0004      /* unused */
#define	NBF_UPCALLED	0x0010      /* unused */
#define	NBF_NETBIOS		0x0020
#define NBF_BOUND_IF    0x0040
#define NBF_SOCK_OPENED 0x0080      /* socket opened by sock_socket */


/*
 * socket specific data
 */
struct nbpcb {
	struct smbiod      *nbp_iod;
	socket_t            nbp_tso;    /* transport socket */
	struct sockaddr_nb *nbp_laddr;  /* local address */
	struct sockaddr_nb *nbp_paddr;  /* peer address */

	int                 nbp_flags;
	enum nbstate        nbp_state;
	struct timespec     nbp_timo;
	uint32_t            nbp_sndbuf;
	uint32_t            nbp_rcvbuf;
	uint32_t            nbp_rcvchunk;
	void               *nbp_selectid;
	void              (*nbp_upcall)(void *);
	uint32_t            nbp_qos;
    struct sockaddr_storage nbp_sock_addr;
    uint32_t            nbp_if_idx;
/*	LIST_ENTRY(nbpcb) nbp_link;*/
};

/*
 * TCP slowstart presents a problem in conjunction with large
 * reads.  To ensure a steady stream of ACKs while reading using
 * large transaction sizes, we call soreceive() with a smaller
 * buffer size.  See nbssn_recv().
 */
#define NB_SORECEIVE_CHUNK	(8 * 1024)

extern struct smb_tran_desc smb_tran_nbtcp_desc;

#define SMBSBTIMO		5 /* seconds for sockbuf timeouts */
#define SMB_SB_RCVTIMEO 15 /* seconds before we give up on a sock_receivembuf */

#endif /* _KERNEL */

#endif /* !_NETSMB_SMB_TRANTCP_H_ */
