/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 21, 2022.
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
/* TCP-cache to store and retrieve TCP-related information */

#ifndef _NETINET_TCP_CACHE_H
#define _NETINET_TCP_CACHE_H

#include <netinet/tcp_var.h>
#include <netinet/in.h>

#define ECN_MIN_CE_PROBES       10 /* Probes are basically the number of incoming packets */
#define ECN_MAX_CE_RATIO        7 /* Ratio is the maximum number of CE-packets we accept per incoming "probe" */

extern void tcp_cache_set_cookie(struct tcpcb *tp, u_char *__counted_by(len) cookie, u_int8_t len);
extern int tcp_cache_get_cookie(struct tcpcb *tp, u_char *__counted_by(buflen) cookie, uint8_t buflen, u_int8_t *len);
extern unsigned int tcp_cache_get_cookie_len(struct tcpcb *tp);
extern uint8_t tcp_cache_get_mptcp_version(struct sockaddr* dst);
extern void tcp_cache_update_mptcp_version(struct tcpcb *tp, boolean_t succeeded);

extern void tcp_heuristic_tfo_loss(struct tcpcb *tp);
extern void tcp_heuristic_tfo_rst(struct tcpcb *tp);
extern void tcp_heuristic_mptcp_loss(struct tcpcb *tp);
extern void tcp_heuristic_ecn_loss(struct tcpcb *tp);
extern void tcp_heuristic_tfo_middlebox(struct tcpcb *tp);
extern void tcp_heuristic_ecn_aggressive(struct tcpcb *tp);
extern void tcp_heuristic_tfo_success(struct tcpcb *tp);
extern void tcp_heuristic_mptcp_success(struct tcpcb *tp);
extern void tcp_heuristic_ecn_success(struct tcpcb *tp);
extern boolean_t tcp_heuristic_do_tfo(struct tcpcb *tp);
extern int tcp_heuristic_do_mptcp(struct tcpcb *tp);
extern boolean_t tcp_heuristic_do_ecn(struct tcpcb *tp);
extern void tcp_heuristic_ecn_droprst(struct tcpcb *tp);
extern void tcp_heuristic_ecn_droprxmt(struct tcpcb *tp);
extern void tcp_heuristic_ecn_synrst(struct tcpcb *tp);

extern boolean_t tcp_heuristic_do_ecn_with_address(struct ifnet *ifp,
    union sockaddr_in_4_6 *local_address);
extern void tcp_heuristics_ecn_update(struct necp_tcp_ecn_cache *necp_buffer,
    struct ifnet *ifp, union sockaddr_in_4_6 *local_address);
extern boolean_t tcp_heuristic_do_tfo_with_address(struct ifnet *ifp,
    union sockaddr_in_4_6 *local_address, union sockaddr_in_4_6 *remote_address,
    u_int8_t *__counted_by(maxlen) cookie, u_int8_t maxlen, u_int8_t *cookie_len);
extern void tcp_heuristics_tfo_update(struct necp_tcp_tfo_cache *necp_buffer,
    struct ifnet *ifp, union sockaddr_in_4_6 *local_address,
    union sockaddr_in_4_6 *remote_address);

extern void tcp_cache_init(void);

#endif /* _NETINET_TCP_CACHE_H */
