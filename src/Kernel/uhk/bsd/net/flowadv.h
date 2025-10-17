/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 4, 2025.
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
#ifndef _NET_FLOWADV_H_
#define _NET_FLOWADV_H_

#ifdef KERNEL_PRIVATE
#include <sys/types.h>
#include <sys/queue.h>

#if SKYWALK
#include <skywalk/os_skywalk.h>
#endif /* SKYWALK */

#define FADV_SUCCESS            0       /* success */
#define FADV_FLOW_CONTROLLED    1       /* regular flow control */
#define FADV_SUSPENDED          2       /* flow control due to suspension */

struct flowadv {
	int32_t         code;           /* FADV advisory code */
};

typedef enum fce_event_type {
	FCE_EVENT_TYPE_FLOW_CONTROL_FEEDBACK   = 0,
	FCE_EVENT_TYPE_CONGESTION_EXPERIENCED  = 1,
} fce_event_type_t;

#ifdef BSD_KERNEL_PRIVATE
struct flowadv_fcentry {
	STAILQ_ENTRY(flowadv_fcentry) fce_link;
	u_int32_t        fce_flowsrc_type;       /* FLOWSRC values */
	u_int32_t        fce_flowid;
	u_int32_t        fce_ce_cnt;
	u_int32_t        fce_pkts_since_last_report;
	fce_event_type_t fce_event_type;
#if SKYWALK
	flowadv_token_t fce_flowsrc_token;
	flowadv_idx_t   fce_flowsrc_fidx;
	struct ifnet    *fce_ifp;
#endif /* SKYWALK */
};

STAILQ_HEAD(flowadv_fclist, flowadv_fcentry);

__BEGIN_DECLS

extern void flowadv_init(void);
extern struct flowadv_fcentry *flowadv_alloc_entry(int);
extern void flowadv_free_entry(struct flowadv_fcentry *);
extern void flowadv_add(struct flowadv_fclist *);
extern void flowadv_add_entry(struct flowadv_fcentry *);

__END_DECLS

#endif /* BSD_KERNEL_PRIVATE */
#endif /* KERNEL_PRIVATE */
#endif /* _NET_FLOWADV_H_ */
