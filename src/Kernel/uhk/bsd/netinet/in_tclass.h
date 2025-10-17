/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 16, 2022.
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
#ifndef __NETINET_IN_TCLASS_H__
#define __NETINET_IN_TCLASS_H__

#ifdef PRIVATE

#include <sys/types.h>
#include <sys/socket.h>
#include <sys/socketvar.h>
#include <sys/mbuf.h>
#include <net/if.h>
#include <net/if_var.h>

#define SO_TCDBG_PID            0x01    /* Set/get traffic class policy for PID */
#define SO_TCDBG_PNAME          0x02    /* Set/get traffic class policy for processes of that name */
#define SO_TCDBG_PURGE          0x04    /* Purge entries for unused PIDs */
#define SO_TCDBG_FLUSH          0x08    /* Flush all entries */
#define SO_TCDBG_COUNT          0x10    /* Get count of entries */
#define SO_TCDBG_LIST           0x20    /* List entries */
#define SO_TCDBG_DELETE         0x40    /* Delete a process entry */
#define SO_TCDBG_TCFLUSH_PID    0x80    /* Flush traffic class for PID */

struct so_tcdbg {
	u_int32_t       so_tcdbg_cmd;
	int32_t         so_tcdbg_tclass;
	int32_t         so_tcdbg_netsvctype;
	uint8_t         so_tcdbg_ecn_val;  /* 1 is ECT(1) and 2 is ECT(0) */
	u_int32_t       so_tcdbg_count;
	pid_t           so_tcdbg_pid;
	u_int32_t       so_tcbbg_qos_mode;
	char            so_tcdbg_pname[(2 * MAXCOMLEN) + 1];
};
#define QOS_MODE_MARKING_POLICY_DEFAULT         0
#define QOS_MODE_MARKING_POLICY_ENABLE          1
#define QOS_MODE_MARKING_POLICY_DISABLE         2

#define NET_QOS_MARKING_POLICY_DEFAULT QOS_MODE_MARKING_POLICY_DEFAULT /* obsolete, to be removed */
#define NET_QOS_MARKING_POLICY_ENABLE QOS_MODE_MARKING_POLICY_ENABLE /* obsolete, to be removed */
#define NET_QOS_MARKING_POLICY_DISABLE QOS_MODE_MARKING_POLICY_DISABLE /* obsolete, to be removed */

struct net_qos_param {
	u_int64_t nq_transfer_size;     /* transfer size in bytes */
	u_int32_t nq_use_expensive:1,   /* allowed = 1 otherwise 0 */
	    nq_uplink:1,                /* uplink = 1 otherwise 0 */
	    nq_use_constrained:1;       /* allowed = 1 otherwise 0 */
	u_int32_t nq_unused;            /* for future expansion */
};

#ifndef KERNEL

/*
 * Returns whether a large upload or download transfer should be marked as
 * BK service type for network activity. This is a system level
 * hint/suggestion to classify application traffic based on statistics
 * collected from the current network attachment
 *
 *	@param	param	transfer parameters
 *	@param	param_len parameter length
 *	@return	returns 1 for BK and 0 for default
 */
extern int net_qos_guideline(struct net_qos_param *param, size_t param_len);

#endif /* !KERNEL */

#ifdef BSD_KERNEL_PRIVATE

extern int net_qos_policy_restricted;
extern int net_qos_policy_wifi_enabled;
extern int net_qos_policy_capable_enabled;

extern void net_qos_map_init(void);
extern void net_qos_map_change(uint32_t mode);
extern errno_t set_packet_qos(struct mbuf *, struct ifnet *, boolean_t, int,
    int, u_int8_t *);
extern int so_get_netsvc_marking_level(struct socket *);

extern uint8_t fastlane_sc_to_dscp(uint32_t svc_class);
extern uint8_t rfc4594_sc_to_dscp(uint32_t svc_class);
extern uint8_t custom_sc_to_dscp(uint32_t svc_class);

extern mbuf_traffic_class_t rfc4594_dscp_to_tc(uint8_t dscp);

#endif /* BSD_KERNEL_PRIVATE */

#endif /* PRIVATE */

#endif /* __NETINET_IN_TCLASS_H__ */
