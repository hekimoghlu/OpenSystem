/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 23, 2023.
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
#ifndef _NETINET_TCP_UTILS_H_
#define _NETINET_TCP_UTILS_H_

#include <netinet/tcp_var.h>

struct tcp_globals {};

static inline struct tcp_globals *
tcp_get_globals(struct tcpcb *tp)
{
#pragma unused(tp)
	return NULL;
}

static inline uint32_t
tcp_globals_now(struct tcp_globals *globals)
{
#pragma unused(globals)
	return tcp_now;
}

extern void tcp_ccdbg_control_register(void);
extern void tcp_ccdbg_trace(struct tcpcb *tp, struct tcphdr *th, int32_t event);

#endif /* _NETINET_TCP_UTILS_H_ */
