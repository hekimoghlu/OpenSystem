/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 12, 2021.
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
/*
 * Copyright (c) 1982, 1986, 1993
 *	The Regents of the University of California.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 3. All advertising materials mentioning features or use of this software
 *    must display the following acknowledgement:
 *	This product includes software developed by the University of
 *	California, Berkeley and its contributors.
 * 4. Neither the name of the University nor the names of its contributors
 *    may be used to endorse or promote products derived from this software
 *    without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE REGENTS AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 *
 *	@(#)udp.h	8.1 (Berkeley) 6/10/93
 */

#ifndef _NETINET_UDP_H_
#define _NETINET_UDP_H_

#ifndef DRIVERKIT
#include <sys/appleapiopts.h>
#include <sys/types.h>          /* u_short */
#else
#include <sys/_types.h>
#include <sys/_types/_u_char.h>
#include <sys/_types/_u_short.h>
#endif /* DRIVERKIT */

/*
 * Udp protocol header.
 * Per RFC 768, September, 1981.
 */
struct udphdr {
	u_short uh_sport;               /* source port */
	u_short uh_dport;               /* destination port */
	u_short uh_ulen;                /* udp length */
	u_short uh_sum;                 /* udp checksum */
};

/*
 * User-settable options (used with setsockopt).
 */
#define UDP_NOCKSUM     0x01    /* don't checksum outbound payloads */
#ifdef PRIVATE
#define UDP_KEEPALIVE_OFFLOAD   0x02 /* Send keep-alive at a given interval */
#endif /* PRIVATE */

#ifdef PRIVATE
/*
 * This is a mechanism to offload keep-alive or heartbeat messages
 * to the Wifi driver when the host processor is sleeping. The application
 * will give a small amount of data that can be placed in the message. The
 * application will also specify an interval at which these messages
 * should be sent.
 *
 * The purpose of these messages is to detect loss of connectivity in
 * peer-to-peer communication without keeping the host processor awake.
 *
 * The application will pass this data to the kernel using setsockopt. It
 * can set the interval to 0 to disable keepalive offload.
 */
#define UDP_KEEPALIVE_OFFLOAD_DATA_SIZE 32

/* Maximum keep alive interval in seconds */
#define UDP_KEEPALIVE_INTERVAL_MAX_SECONDS      65536

struct udp_keepalive_offload {
	u_char ka_data[UDP_KEEPALIVE_OFFLOAD_DATA_SIZE];
	u_int16_t ka_interval;          /* interval in seconds */
	u_int8_t ka_data_len;           /* valid length of ka_data */
	u_int8_t ka_type;               /* type of application */
#define UDP_KEEPALIVE_OFFLOAD_TYPE_AIRPLAY      0x1
};

#endif /* PRIVATE */
#endif /* _NETINET_UDP_H */
