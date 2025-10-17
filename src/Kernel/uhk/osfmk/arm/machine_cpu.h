/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 18, 2025.
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
#ifndef _ARM_MACHINE_CPU_H_
#define _ARM_MACHINE_CPU_H_

#include <mach/mach_types.h>
#include <mach/boolean.h>
#include <kern/kern_types.h>
#include <pexpert/pexpert.h>
#include <arm/cpu_data_internal.h>

extern void cpu_machine_init(void);

extern kern_return_t cpu_register(int *slot_nump);

extern void cpu_signal_handler(void);
extern void cpu_signal_handler_internal(boolean_t disable_signal);

extern void cpu_doshutdown(void (*doshutdown)(processor_t), processor_t processor);

extern void cpu_idle(void);
extern void cpu_idle_exit(boolean_t from_reset) __attribute__((noreturn));
extern void cpu_idle_tickle(void);

extern void cpu_machine_idle_init(boolean_t from_boot);

extern void arm_init_cpu(cpu_data_t *args, uint64_t hib_header_phys);

extern void arm_init_idle_cpu(cpu_data_t *args);

extern void init_cpu_timebase(boolean_t enable_fiq);

#define cpu_pause() do {} while (0)     /* Not for this architecture */
bool
wfe_to_deadline_or_interrupt(uint32_t cid, uint64_t wfe_deadline, cpu_data_t *cdp, bool unmask, bool check_cluster_recommendation);
#endif /* _ARM_MACHINE_CPU_H_ */
