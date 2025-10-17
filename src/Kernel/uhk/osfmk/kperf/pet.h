/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 14, 2024.
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
#include <kern/thread.h>
#include <stdbool.h>

#define KPERF_PET_DEFAULT_IDLE_RATE 15

extern uint64_t kppet_lightweight_start_time;
extern _Atomic uint32_t kppet_gencount;

/*
 * If `actionid` is non-zero, set up PET to sample the action.  Otherwise,
 * disable PET.
 */
void kppet_config(unsigned int actionid);

/*
 * Reset PET back to its default settings.
 */
void kppet_reset(void);

/*
 * Notify PET that new threads are switching on-CPU.
 */
void kppet_on_cpu(thread_t thread, thread_continue_t continuation,
    uintptr_t *starting_frame);

/*
 * Mark a thread as sampled by PET.
 */
void kppet_mark_sampled(thread_t thread);

/*
 * Wake the PET thread from its timer handler.
 */
void kppet_wake_thread(void);

/*
 * For configuring PET from the sysctl interface.
 */
int kppet_get_idle_rate(void);
int kppet_set_idle_rate(int new_idle_rate);
int kppet_get_lightweight_pet(void);
int kppet_set_lightweight_pet(int on);

/*
 * Update whether lightweight PET is active when turning sampling on and off.
 */
void kppet_lightweight_active_update(void);

/*
 * Notify from kptimer what the PET period is.
 */
void kppet_set_period(uint64_t period);
