/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 19, 2023.
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
#ifndef _ARM_LOCKS_H_
#define _ARM_LOCKS_H_

#ifdef  MACH_KERNEL_PRIVATE
#ifndef LCK_SPIN_IS_TICKET_LOCK
#define LCK_SPIN_IS_TICKET_LOCK 0
#endif
#endif /* MACH_KERNEL_PRIVATE */

#include <kern/lock_types.h>
#ifdef  MACH_KERNEL_PRIVATE
#include <kern/sched_hygiene.h>
#include <kern/startup.h>
#if LCK_SPIN_IS_TICKET_LOCK
#include <kern/ticket_lock.h>
#endif
#endif

#ifdef  MACH_KERNEL_PRIVATE
#if LCK_SPIN_IS_TICKET_LOCK
typedef lck_ticket_t lck_spin_t;
#else
typedef struct lck_spin_s {
	struct hslock   hwlock;
	unsigned long   type;
} lck_spin_t;

#define lck_spin_data hwlock.lock_data

#define LCK_SPIN_TAG_DESTROYED  0xdead  /* lock marked as Destroyed */

#define LCK_SPIN_TYPE           0x00000011
#define LCK_SPIN_TYPE_DESTROYED 0x000000ee
#endif

#elif KERNEL_PRIVATE

typedef struct {
	uintptr_t opaque[2] __kernel_data_semantics;
} lck_spin_t;

typedef struct {
	uintptr_t opaque[2] __kernel_data_semantics;
} lck_mtx_t;

typedef struct {
	uintptr_t opaque[16];
} lck_mtx_ext_t;

#else

typedef struct __lck_spin_t__           lck_spin_t;
typedef struct __lck_mtx_t__            lck_mtx_t;
typedef struct __lck_mtx_ext_t__        lck_mtx_ext_t;

#endif  /* !KERNEL_PRIVATE */
#ifdef  MACH_KERNEL_PRIVATE

/*
 * static panic deadline, in timebase units, for
 * hw_lock_{bit,lock}{,_nopreempt} and hw_wait_while_equals()
 */
extern uint64_t _Atomic lock_panic_timeout;

/* Adaptive spin before blocking */
extern uint64_t            MutexSpin;
extern uint64_t            low_MutexSpin;
extern int64_t             high_MutexSpin;

#if CONFIG_PV_TICKET
extern bool                has_lock_pv;
#endif

#ifdef LOCK_PRIVATE

#define LOCK_SNOOP_SPINS        100
#define LOCK_PRETEST            0

#define wait_for_event()        __builtin_arm_wfe()

#if SCHED_HYGIENE_DEBUG
#define lock_disable_preemption_for_thread(t) ({                                \
	thread_t __dpft_thread = (t);                                           \
	uint32_t *__dpft_countp = &__dpft_thread->machine.preemption_count;     \
	uint32_t __dpft_count;                                                  \
                                                                                \
	__dpft_count = *__dpft_countp;                                          \
	os_atomic_store(__dpft_countp, __dpft_count + 1, compiler_acq_rel);     \
                                                                                \
	if (static_if(sched_debug_preemption_disable)) {                        \
	       if (__dpft_count == 0 && sched_preemption_disable_debug_mode) {  \
	               _prepare_preemption_disable_measurement();               \
	       }                                                                \
	}                                                                       \
})
#else /* SCHED_HYGIENE_DEBUG */
#define lock_disable_preemption_for_thread(t) ({                                \
	uint32_t *__dpft_countp = &(t)->machine.preemption_count;               \
                                                                                \
	os_atomic_store(__dpft_countp, *__dpft_countp + 1, compiler_acq_rel);   \
})
#endif /* SCHED_HYGIENE_DEBUG */
#define lock_enable_preemption()                enable_preemption()
#define lock_preemption_level_for_thread(t)     get_preemption_level_for_thread(t)
#define lock_preemption_disabled_for_thread(t)  (get_preemption_level_for_thread(t) != 0)
#define current_thread()                        current_thread_fast()

#define __hw_spin_wait_load(ptr, load_var, cond_result, cond_expr) ({ \
	load_var = os_atomic_load_exclusive(ptr, relaxed);                      \
	cond_result = (cond_expr);                                              \
	if (__probable(cond_result)) {                                          \
	        os_atomic_clear_exclusive();                                    \
	} else {                                                                \
	        wait_for_event();                                               \
	}                                                                       \
})

#endif /* LOCK_PRIVATE */
#endif /* MACH_KERNEL_PRIVATE */
#endif /* _ARM_LOCKS_H_ */
