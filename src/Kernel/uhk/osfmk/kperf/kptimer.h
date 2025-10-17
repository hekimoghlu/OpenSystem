/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 5, 2023.
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
#ifndef KPERF_KPTIMER_H
#define KPERF_KPTIMER_H

/*
 * kptimer is responsible for managing the kperf's on-CPU timers.  These
 * timers sample threads that are running on CPUs at a cadence determined by a
 * specified period.  When they fire, a handler runs the specified action and
 * reprograms the timer to fire again.  To get everything started or stopped,
 * kptimer issues a broadcast IPI to modify kperf's multiplexed per-CPU timer,
 * stored in the machine-dependent per-CPU structure.
 *
 * On-CPU timers are disabled when the CPU they've been programmed for goes idle
 * to prevent waking up the idle CPU when it's not running anything interesting.
 * This logic lives in the platform code that's responsible for entering and
 * exiting idle.
 *
 * Traditional PET is configured here (since it's defined by identifying a timer
 * to use for PET) but its mechanism is in osfmk/kperf/pet.c.  Lightweight PET
 * does use kptimer to increment its generation count, however.
 */

/*
 * The minimum allowed timer period depends on the type of client (foreground vs.
 * background) and timer (on-CPU vs. PET).
 */
enum kptimer_period_limit {
	KTPL_FG,
	KTPL_BG,
	KTPL_FG_PET,
	KTPL_BG_PET,
	KTPL_MAX,
};

/*
 * The minimum timer periods allowed by kperf.  There's no other mechanism
 * to prevent interrupt storms due to kptimer.
 */
extern const uint64_t kptimer_minperiods_ns[KTPL_MAX];

/*
 * Called from the kernel startup thread to set up kptimer.
 */
void kptimer_init(void);

/*
 * Return the minimum timer period in Mach time units.
 */
uint64_t kptimer_min_period_abs(bool pet);

/*
 * Return the number of timers available.
 */
unsigned int kptimer_get_count(void);

/*
 * Set the number of timers available to `count`.
 *
 * Returns 0 on success, and non-0 on error.
 */
int kptimer_set_count(unsigned int count);

/*
 * Return the period of the timer identified by `timerid` in `period_out`.
 *
 * Returns 0 on success, and non-0 on error.
 */
int kptimer_get_period(unsigned int timerid, uint64_t *period_out);

/*
 * Set the period of the timer identified by `timerid` to `period`.
 *
 * Returns non-zero on error, and zero otherwise.
 */
int kptimer_set_period(unsigned int timerid, uint64_t period);

/*
 * Return the action of the timer identified by `timerid` in
 * `actionid_out`.
 */
int kptimer_get_action(unsigned int timerid, uint32_t *actionid_out);

/*
 * Set the action of the timer identified by `timerid` to `actionid`.
 */
int kptimer_set_action(unsigned int timer, uint32_t actionid);

/*
 * Set the PET timer to the timer identified by `timerid`.
 */
int kptimer_set_pet_timerid(unsigned int timerid);

/*
 * Return the ID of the PET timer.
 */
unsigned int kptimer_get_pet_timerid(void);

/*
 * For PET to rearm its timer after its sampling thread took `sampledur_abs`
 * to sample.
 */
void kptimer_pet_enter(uint64_t sampledur_abs);

/*
 * Start all active timers.  The ktrace lock must be held.
 */
void kptimer_start(void);

/*
 * Stop all active timers, waiting for them to stop.  The ktrace lock must be held.
 */
void kptimer_stop(void);

/*
 * Cancel outstanding kperf timer for this CPU.
 */
void kptimer_stop_curcpu(void);

/*
 * Reconfigure this CPU's kptimer expiration when it's brought online
 */
void kptimer_curcpu_up(void);

/*
 * To indicate the next timer has expired.
 */
void kptimer_expire(processor_t processor, int cpuid, uint64_t now);

/*
 * Reset the kptimer system.
 */
void kptimer_reset(void);

#endif /* !defined(KPERF_KPTIMER_H) */
