/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 8, 2025.
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
 *
 */

#include <stdbool.h>
#include <stdint.h>

#ifndef KERN_SCHED_HYGIENE_DEBUG
#define KERN_SCHED_HYGIENE_DEBUG

#if SCHED_HYGIENE_DEBUG

#if !DEVELOPMENT && !DEBUG
#error SCHED_HYGIENE_DEBUG defined without DEVELOPMENT/DEBUG
#endif

#include <mach/mach_types.h>
#include <machine/static_if.h>
#include <kern/startup.h>

typedef enum sched_hygiene_mode {
	SCHED_HYGIENE_MODE_OFF = 0,
	SCHED_HYGIENE_MODE_TRACE = 1,
	SCHED_HYGIENE_MODE_PANIC = 2,
} sched_hygiene_mode_t;

STATIC_IF_KEY_DECLARE_TRUE(sched_debug_pmc);
STATIC_IF_KEY_DECLARE_TRUE(sched_debug_preemption_disable);
/* implies sched_debug_preemption_disable */
STATIC_IF_KEY_DECLARE_TRUE(sched_debug_interrupt_disable);

extern sched_hygiene_mode_t sched_preemption_disable_debug_mode;

MACHINE_TIMEOUT_SPEC_DECL(sched_preemption_disable_threshold_mt);
extern machine_timeout_t sched_preemption_disable_threshold_mt;

__attribute__((noinline)) void _prepare_preemption_disable_measurement(void);
__attribute__((noinline)) void _collect_preemption_disable_measurement(void);

extern sched_hygiene_mode_t interrupt_masked_debug_mode;

MACHINE_TIMEOUT_SPEC_DECL(interrupt_masked_timeout);
MACHINE_TIMEOUT_SPEC_DECL(stackshot_interrupt_masked_timeout);

extern machine_timeout_t interrupt_masked_timeout;
extern machine_timeout_t stackshot_interrupt_masked_timeout;

extern bool sched_hygiene_nonspec_tb;

#define ml_get_sched_hygiene_timebase() (sched_hygiene_nonspec_tb ? ml_get_timebase() : ml_get_speculative_timebase())

extern bool kprintf_spam_mt_pred(struct machine_timeout_spec const *spec);

#endif /* SCHED_HYGIENE_DEBUG */

#endif /* KERN_SCHED_HYGIENE_DEBUG */
