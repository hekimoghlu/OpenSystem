/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 3, 2024.
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
#ifndef __KPI_IPFILTER_VAR__
#define __KPI_IPFILTER_VAR__

#include <sys/appleapiopts.h>
#include <netinet/kpi_ipfilter.h>

#ifdef KERNEL_PRIVATE

/* Private data structure, stripped out by ifdef tool */
/* Implementation specific bits */

#include <sys/queue.h>

struct ipfilter {
	TAILQ_ENTRY(ipfilter)   ipf_link;
	struct ipf_filter       ipf_filter;
	struct ipfilter_list    *ipf_head;
	TAILQ_ENTRY(ipfilter)   ipf_tbr;
	uint32_t                ipf_flags;
};
TAILQ_HEAD(ipfilter_list, ipfilter);

#define IPFF_INTERNAL 0x1

extern struct ipfilter_list     ipv6_filters;
extern struct ipfilter_list     ipv4_filters;

extern ipfilter_t ipf_get_inject_filter(struct mbuf *m);
extern void ipf_ref(void);
extern void ipf_unref(void);
extern void ip_proto_dispatch_in(struct mbuf *m, int hlen, u_int8_t proto,
    ipfilter_t ipfref);

extern void ipfilter_register_m_tag(void);

#endif /* KERNEL_PRIVATE */

#endif /*__KPI_IPFILTER_VAR__*/
