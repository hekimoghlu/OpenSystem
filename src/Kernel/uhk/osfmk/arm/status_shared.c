/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 5, 2025.
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
#include <debug.h>
#include <mach/mach_types.h>
#include <mach/kern_return.h>
#include <mach/thread_status.h>
#include <kern/thread.h>
#include <arm/vmparam.h>
#include <arm/cpu_data_internal.h>

/*
 * Copy values from saved_state to ts32.
 */
void
saved_state_to_thread_state32(const arm_saved_state_t *saved_state, arm_thread_state32_t *ts32)
{
	uint32_t i;

	assert(is_saved_state32(saved_state));

	ts32->lr = (uint32_t)get_saved_state_lr(saved_state);
	ts32->sp = (uint32_t)get_saved_state_sp(saved_state);
	ts32->pc = (uint32_t)get_saved_state_pc(saved_state);
	ts32->cpsr = get_saved_state_cpsr(saved_state);
	for (i = 0; i < 13; i++) {
		ts32->r[i] = (uint32_t)get_saved_state_reg(saved_state, i);
	}
}

/*
 * Copy values from ts32 to saved_state.
 */
void
thread_state32_to_saved_state(const arm_thread_state32_t *ts32, arm_saved_state_t *saved_state)
{
	uint32_t i;

	assert(is_saved_state32(saved_state));

	set_user_saved_state_lr(saved_state, ts32->lr);
	set_saved_state_sp(saved_state, ts32->sp);
	set_user_saved_state_pc(saved_state, ts32->pc);

#if defined(__arm64__)
	set_user_saved_state_cpsr(saved_state, (ts32->cpsr & ~PSR64_MODE_MASK) | PSR64_MODE_RW_32);
#else
#error Unknown architecture.
#endif

	for (i = 0; i < 13; i++) {
		set_user_saved_state_reg(saved_state, i, ts32->r[i]);
	}
}
