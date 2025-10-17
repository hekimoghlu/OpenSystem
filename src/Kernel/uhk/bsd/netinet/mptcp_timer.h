/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 30, 2024.
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
#ifndef _NETINET_MPTCP_TIMER_H_
#define _NETINET_MPTCP_TIMER_H_

#ifdef BSD_KERNEL_PRIVATE

__BEGIN_DECLS
extern uint32_t mptcp_timer(struct mppcbinfo *mppi);
extern void mptcp_start_timer(struct mptses *mpte, int timer_type);
extern void mptcp_cancel_timer(struct mptcb *mp_tp, int timer_type);
extern void mptcp_cancel_all_timers(struct mptcb *mp_tp);
extern void mptcp_init_urgency_timer(struct mptses *mpte);
extern void mptcp_set_urgency_timer(struct mptses *mpte);
__END_DECLS

#endif /* BSD_KERNEL_PRIVATE */
#endif /* !_NETINET_MPTCP_TIMER_H_ */
