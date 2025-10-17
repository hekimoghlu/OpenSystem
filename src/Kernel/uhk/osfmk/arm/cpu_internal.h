/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 9, 2025.
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
 * @OSF_COPYRIGHT@
 */
#ifndef _ARM_CPU_INTERNAL_H_
#define _ARM_CPU_INTERNAL_H_


#include <mach/kern_return.h>
#include <arm/cpu_data_internal.h>

extern void                                             cpu_bootstrap(
	void);

extern void                                             cpu_init(
	void);

extern void                                             cpu_timebase_init(boolean_t from_boot);

extern kern_return_t                    cpu_signal(
	cpu_data_t              *target,
	cpu_signal_t    signal,
	void                    *p0,
	void                    *p1);

extern kern_return_t                    cpu_signal_deferred(
	cpu_data_t              *target);

extern void                     cpu_signal_cancel(
	cpu_data_t              *target);

extern bool cpu_has_SIGPdebug_pending(void);

extern unsigned int real_ncpus;

#if defined(CONFIG_XNUPOST) && __arm64__
extern void arm64_ipi_test(void);
#endif /* defined(CONFIG_XNUPOST) && __arm64__ */

#if defined(KERNEL_INTEGRITY_CTRR)
extern void init_ctrr_cluster_states(void);
extern lck_spin_t ctrr_cpu_start_lck;
enum ctrr_cluster_states { CTRR_UNLOCKED = 0, CTRR_LOCKING, CTRR_LOCKED };
extern enum ctrr_cluster_states ctrr_cluster_locked[MAX_CPU_CLUSTERS];
#endif /* defined(KERNEL_INTEGRITY_CTRR) */

#endif  /* _ARM_CPU_INTERNAL_H_ */
