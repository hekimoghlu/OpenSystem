/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 1, 2025.
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
 *	@(#)icmp_var.h	8.1 (Berkeley) 6/10/93
 * $FreeBSD: src/sys/netinet/icmp_var.h,v 1.15.2.1 2001/02/24 21:35:18 bmilekic Exp $
 */

#ifndef _NETINET_ICMP_VAR_H_
#define _NETINET_ICMP_VAR_H_
#include <sys/appleapiopts.h>

#include <netinet/ip_icmp.h>
#include <sys/types.h>

/*
 * Variables related to this implementation
 * of the internet control message protocol.
 */
struct  icmpstat {
/* statistics related to icmp packets generated */
	u_int32_t       icps_error;     /* # of calls to icmp_error */
	u_int32_t       icps_oldshort;  /* no error 'cuz old ip too short */
	u_int32_t       icps_oldicmp;   /* no error 'cuz old was icmp */
	u_int32_t       icps_outhist[ICMP_MAXTYPE + 1];
/* statistics related to input messages processed */
	u_int32_t       icps_badcode;   /* icmp_code out of range */
	u_int32_t       icps_tooshort;  /* packet < ICMP_MINLEN */
	u_int32_t       icps_checksum;  /* bad checksum */
	u_int32_t       icps_badlen;    /* calculated bound mismatch */
	u_int32_t       icps_reflect;   /* number of responses */
	u_int32_t       icps_inhist[ICMP_MAXTYPE + 1];
	u_int32_t       icps_bmcastecho;/* b/mcast echo requests dropped */
	u_int32_t       icps_bmcasttstamp; /* b/mcast tstamp requests dropped */
};

/*
 * Names for ICMP sysctl objects
 */
#define ICMPCTL_MASKREPL        1       /* allow replies to netmask requests */
#define ICMPCTL_STATS           2       /* statistics (read-only) */
#define ICMPCTL_ICMPLIM         3
#define ICMPCTL_TIMESTAMP       4       /* allow replies to time stamp requests */
#define ICMPCTL_ICMPLIM_INCR    5
#define ICMPCTL_MAXID           6

#ifdef BSD_KERNEL_PRIVATE
#define ICMPCTL_NAMES { \
	{ 0, 0 }, \
	{ "maskrepl", CTLTYPE_INT }, \
	{ "stats", CTLTYPE_STRUCT }, \
	{ "icmplim", CTLTYPE_INT }, \
	{ "icmptimestamp", CTLTYPE_INT }, \
}

SYSCTL_DECL(_net_inet_icmp);
extern struct   icmpstat icmpstat;
#endif /* BSD_KERNEL_PRIVATE */
#endif /* _NETINET_ICMP_VAR_H_ */
