/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 20, 2022.
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
 * Copyright (c) 1992, 1993
 *	Regents of the University of California.  All rights reserved.
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
 *	@(#)netstat.h	8.2 (Berkeley) 1/4/94
 */

#include <sys/cdefs.h>
#include <sys/types.h>
#include <stdint.h>

#include <TargetConditionals.h>

extern int	Aflag;	/* show addresses of protocol control block */
extern int	aflag;	/* show all sockets (including servers) */
extern int	bflag;	/* show i/f total bytes in/out */
extern int	cflag;	/* show specific classq */
extern int	dflag;	/* show i/f dropped packets */
extern int	Fflag;	/* show i/f forwarded packets */
extern int	gflag;	/* show group (multicast) routing or stats */
extern int	iflag;	/* show interfaces */
extern int	lflag;	/* show routing table with use and ref */
extern int	Lflag;	/* show size of listen queues */
extern int	mflag;	/* show memory stats */
extern int	nflag;	/* show addresses numerically */
extern int	Rflag;	/* show reachability information */
extern int	rflag;	/* show routing tables (or routing stats) */
extern int	sflag;	/* show protocol statistics */
extern int	prioflag; /* show packet priority  statistics */
extern int	tflag;	/* show i/f watchdog timers */
extern int	vflag;	/* more verbose */
extern int	Wflag;	/* wide display */
extern int	qflag;	/* Display ifclassq stats */
extern int	Qflag;	/* Display opportunistic polling stats */
extern int	xflag;	/* show extended link-layer reachability information */
extern int	zflag;	/* show only entries with non zero rtt metrics */

extern int	cq;	/* send classq index (-1 for all) */
extern int	interval; /* repeat interval for i/f stats */

extern char	*interface; /* desired i/f for stats, or NULL for all i/fs */
extern int	unit;	/* unit number for above */

extern int	af;	/* address family */

extern char	*plural(int);
extern char	*plurales(int);
extern char	*pluralies(int);

extern void	protopr(uint32_t, char *, int);
extern void	mptcppr(uint32_t, char *, int);
extern void	tcp_stats(uint32_t, char *, int);
extern void	mptcp_stats(uint32_t, char *, int);
extern void	udp_stats(uint32_t, char *, int);
extern void	ip_stats(uint32_t, char *, int);
extern void	icmp_stats(uint32_t, char *, int);
extern void	igmp_stats(uint32_t, char *, int);
extern void	arp_stats(uint32_t, char *, int);
#ifdef IPSEC
extern void	ipsec_stats(uint32_t, char *, int);
#endif

extern void tcp_ifstats(char *);
extern void udp_ifstats(char *);

#ifdef INET6
extern void	ip6_stats(uint32_t, char *, int);
extern void	ip6_ifstats(char *);
extern void	icmp6_stats(uint32_t, char *, int);
extern void	icmp6_ifstats(char *);
extern void	rip6_stats(uint32_t, char *, int);

/* forward references */
struct sockaddr_in6;
struct in6_addr;
struct sockaddr;

extern char	*routename6(struct sockaddr_in6 *);
extern char	*netname6(struct sockaddr_in6 *, struct sockaddr *);
#endif /*INET6*/

#ifdef IPSEC
extern void	pfkey_stats(uint32_t, char *, int);
#endif

extern void	systmpr(uint32_t, char *, int);
extern void	kctl_stats(uint32_t, char *, int);
extern void	kevt_stats(uint32_t, char *, int);

extern void	mbpr(void);

extern void	intpr(void (*)(char *));
extern void	intpr_ri(void (*)(char *));
extern void	intervalpr(void (*)(uint32_t, char *, int), uint32_t,
		    char *, int);

extern void	pr_rthdr(int);
extern void	pr_family(int);
extern void	rt_stats(void);
extern void	upHex(char *);
extern char	*routename(uint32_t);
extern char	*netname(uint32_t, uint32_t);
extern void	routepr(void);

extern void	unixpr(uint32_t, char *, int);
extern void	unixstats(uint32_t, char *, int);
extern void	aqstatpr(void);
extern void	rxpollstatpr(void);
extern void	vsockpr(uint32_t,char *,int);
extern void	vsockstats(uint32_t,char *,int);

extern void	ifmalist_dump(void);

extern int print_time(void);
extern void	print_link_status(const char *);

extern void	print_extbkidle_stats(uint32_t, char *, int);
extern void	print_nstat_stats(uint32_t, char *, int);
extern void	print_net_api_stats(uint32_t, char *, int);
extern void	print_if_ports_used_stats(uint32_t, char *, int);
extern void	print_if_link_heuristics_stats(char *);

extern void bpf_stats(char *);
extern void bpf_help(void);

extern void print_socket_stats_format(void);

struct xsocket_n;
struct xsockbuf_n;
struct xsockstat_n;
extern void print_socket_stats_data(struct xsocket_n *, struct xsockbuf_n *, struct xsockbuf_n *, struct xsockstat_n *);
