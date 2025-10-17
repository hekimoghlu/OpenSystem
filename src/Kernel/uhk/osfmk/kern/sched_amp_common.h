/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 2, 2022.
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
#ifndef _KERN_SCHED_AMP_COMMON_H_
#define _KERN_SCHED_AMP_COMMON_H_

#if __AMP__

/* Routine to initialize processor sets on AMP platforms */
void sched_amp_init(void);

/*
 * The AMP scheduler uses spill/steal/rebalance logic to make sure the most appropriate threads
 * are scheduled on the P/E clusters. Here are the definitions of those terms:
 *
 * - Spill:     Spill threads from an overcommited P-cluster onto the E-cluster. This is needed to make sure
 *              that high priority P-recommended threads experience low scheduling latency in the presence of
 *              lots of P-recommended threads.
 *
 * - Steal:     From an E-core, steal a thread from the P-cluster to provide low scheduling latency for
 *              P-recommended threads.
 *
 * - Rebalance: Once a P-core goes idle, check if the E-cores are running any P-recommended threads and
 *              bring it back to run on its recommended cluster type.
 */

/* Spill logic */
int sched_amp_spill_threshold(processor_set_t pset);
void pset_signal_spill(processor_set_t pset, int spilled_thread_priority);
bool pset_should_accept_spilled_thread(processor_set_t pset, int spilled_thread_priority);
bool should_spill_to_ecores(processor_set_t nset, thread_t thread);
void sched_amp_check_spill(processor_set_t pset, thread_t thread);

/* Steal logic */
int sched_amp_steal_threshold(processor_set_t pset, bool spill_pending);
bool sched_amp_steal_thread_enabled(processor_set_t pset);

/* Rebalance logic */
bool sched_amp_balance(processor_t cprocessor, processor_set_t cpset);

/* IPI policy */
sched_ipi_type_t sched_amp_ipi_policy(processor_t dst, thread_t thread, boolean_t dst_idle, sched_ipi_event_t event);

uint32_t sched_amp_qos_max_parallelism(int qos, uint64_t options);
void sched_amp_bounce_thread_group_from_ecores(processor_set_t pset, struct thread_group *tg);

pset_node_t sched_amp_choose_node(thread_t thread);

#endif /* __AMP__ */

#endif /* _KERN_SCHED_AMP_COMMON_H_ */
