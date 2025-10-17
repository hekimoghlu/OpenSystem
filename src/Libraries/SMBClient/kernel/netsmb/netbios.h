/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 3, 2024.
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
#ifndef _NETSMB_NETBIOS_H_
#define	_NETSMB_NETBIOS_H_

/*
 * make this file dirty...
 */
#ifndef _NETINET_IN_H_
#include <netinet/in.h>
#endif

#define PF_NETBIOS	AF_NETBIOS

#define	NBPROTO_TCPSSN	1		/* NETBIOS session over TCP */

#define NB_NAMELEN	16
#define	NB_ENCNAMELEN	NB_NAMELEN * 2
#define	NB_MAXLABLEN	63

#define	NB_MINSALEN	(sizeof(struct sockaddr_nb))

/*
 * name types
 */
#define	NBT_WKSTA	0x00
#define	NBT_CLIENT	0x03
#define	NBT_RASSRVR	0x06
#define	NBT_DMB		0x1B
#define	NBT_IP		0x1C
#define	NBT_MB		0x1D
#define	NBT_BS		0x1E
#define	NBT_NETDDE	0x1F
#define	NBT_SERVER	0x20
#define	NBT_RASCLNT	0x21
#define	NBT_NMAGENT	0xBE
#define	NBT_NMUTIL	0xBF

/*
 * Session packet types
 */
#define	NB_SSN_MESSAGE		0x0
#define	NB_SSN_REQUEST		0x81
#define	NB_SSN_POSRESP		0x82
#define	NB_SSN_NEGRESP		0x83
#define	NB_SSN_RTGRESP		0x84
#define	NB_SSN_KEEPALIVE	0x85

/*
 * resolver: Opcodes
 */
#define	NBNS_OPCODE_QUERY	0x00
#define	NBNS_OPCODE_REGISTER	0x05
#define	NBNS_OPCODE_RELEASE	0x06
#define	NBNS_OPCODE_WACK	0x07
#define	NBNS_OPCODE_REFRESH	0x08
#define	NBNS_OPCODE_RESPONSE	0x10	/* or'ed with other opcodes */

/*
 * resolver: NM_FLAGS
 */
#define	NBNS_NMFLAG_BCAST	0x01
#define	NBNS_NMFLAG_RA		0x08	/* recursion available */
#define	NBNS_NMFLAG_RD		0x10	/* recursion desired */
#define	NBNS_NMFLAG_TC		0x20	/* truncation occured */
#define	NBNS_NMFLAG_AA		0x40	/* authoritative answer */

/* 
 * resolver: Question types
 */
#define	NBNS_QUESTION_TYPE_NB		0x0020
#define NBNS_QUESTION_TYPE_NBSTAT	0x0021

/* 
 * resolver: Question class 
 */
#define NBNS_QUESTION_CLASS_IN	0x0001

/*
 * resolver: Limits
 */
#define	NBNS_MAXREDIRECTS	3	/* maximum number of accepted redirects */
#define	NBDG_MAXSIZE		576	/* maximum nbns datagram size */

/*
 * NETBIOS addressing
 */
struct nb_name {
	unsigned	nn_type;
	u_char		nn_name[NB_NAMELEN + 1];
};

/*
 * Socket address
 */
struct sockaddr_nb {
	u_char				snb_len;
	u_char				snb_family;
	struct sockaddr_in	snb_addrin;
	u_char				snb_name[1 + NB_ENCNAMELEN + 1];	/* encoded */
};

#define GET_IPV4_ADDRESS(a, b) \
	if (a->sa_family == AF_NETBIOS) { \
		b = (struct sockaddr_in *)((void *)&((struct sockaddr_nb *)((void *)a))->snb_addrin); \
	} else if (a->sa_family == AF_INET) {\
		b = (struct sockaddr_in *)((void *)a); \
	} else { \
		b = (struct sockaddr_in *)NULL; \
	}

#endif /* !_NETSMB_NETBIOS_H_ */
