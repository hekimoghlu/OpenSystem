/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 13, 2023.
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
#ifndef _NETINET_MPTCP_OPT_H_
#define _NETINET_MPTCP_OPT_H_

#ifdef BSD_KERNEL_PRIVATE

#include <netinet/tcp.h>
#include <netinet/tcp_var.h>
#include <netinet/mptcp.h>

__BEGIN_DECLS
extern void mptcp_data_ack_rcvd(struct mptcb *mp_tp, struct tcpcb *tp, u_int64_t full_dack);
extern void mptcp_update_window_wakeup(struct tcpcb *tp);
extern void tcp_do_mptcp_options(struct tcpcb *, u_char * opt __ended_by(optend), u_char *optend, struct tcphdr *,
    struct tcpopt *, uint8_t);
extern unsigned mptcp_setup_syn_opts(struct socket *, u_char* __ended_by(optend), u_char *optend, unsigned);
extern unsigned mptcp_setup_join_ack_opts(struct tcpcb *, u_char* __ended_by(optend), u_char *optend, unsigned);
extern unsigned int mptcp_setup_opts(struct tcpcb *tp, int32_t off, u_char *opt __ended_by(optend), u_char *optend,
    unsigned int optlen, int flags, int len,
    boolean_t *p_mptcp_acknow, boolean_t *do_not_compress);
extern void mptcp_update_dss_rcv_state(struct mptcp_dsn_opt *, struct tcpcb *,
    uint16_t);
extern void mptcp_update_rcv_state_meat(struct mptcb *, struct tcpcb *,
    u_int64_t, u_int32_t, u_int16_t, uint16_t);
__END_DECLS

#endif /* BSD_KERNEL_PRIVATE */
#endif /* !_NETINET_MPTCP_OPT_H_ */
