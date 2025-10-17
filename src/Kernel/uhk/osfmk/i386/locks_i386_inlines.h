/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 22, 2024.
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
#ifndef _I386_LOCKS_I386_INLINES_H_
#define _I386_LOCKS_I386_INLINES_H_

#include <kern/locks.h>
#include <kern/lock_stat.h>
#include <kern/turnstile.h>

#if LCK_MTX_USE_ARCH

// Enforce program order of loads and stores.
#define ordered_load(target) os_atomic_load(target, compiler_acq_rel)
#define ordered_store_release(target, value) ({ \
	        os_atomic_store(target, value, release); \
	        os_compiler_barrier(); \
})

/* Enforce program order of loads and stores. */
#define ordered_load_mtx_state(lock)                    ordered_load(&(lock)->lck_mtx_state)
#define ordered_store_mtx_state_release(lock, value)    ordered_store_release(&(lock)->lck_mtx_state, (value))
#define ordered_store_mtx_owner(lock, value)            os_atomic_store(&(lock)->lck_mtx_owner, (value), compiler_acq_rel)

#if DEVELOPMENT | DEBUG
void lck_mtx_owner_check_panic(lck_mtx_t       *mutex) __abortlike;
#endif

__attribute__((always_inline))
static inline void
lck_mtx_ilk_unlock_inline(
	lck_mtx_t       *mutex,
	uint32_t        state)
{
	state &= ~LCK_MTX_ILOCKED_MSK;
	ordered_store_mtx_state_release(mutex, state);

	enable_preemption();
}

__attribute__((always_inline))
static inline void
lck_mtx_lock_finish_inline(
	lck_mtx_t       *mutex,
	uint32_t        state)
{
	assert(state & LCK_MTX_ILOCKED_MSK);

	/* release the interlock and re-enable preemption */
	lck_mtx_ilk_unlock_inline(mutex, state);

	LCK_MTX_ACQUIRED(mutex, mutex->lck_mtx_grp, false,
	    state & LCK_MTX_PROFILE_MSK);
}

__attribute__((always_inline))
static inline void
lck_mtx_lock_finish_inline_with_cleanup(
	lck_mtx_t       *mutex,
	uint32_t        state)
{
	assert(state & LCK_MTX_ILOCKED_MSK);

	/* release the interlock and re-enable preemption */
	lck_mtx_ilk_unlock_inline(mutex, state);

	LCK_MTX_ACQUIRED(mutex, mutex->lck_mtx_grp, false,
	    state & LCK_MTX_PROFILE_MSK);

	turnstile_cleanup();
}

__attribute__((always_inline))
static inline void
lck_mtx_try_lock_finish_inline(
	lck_mtx_t       *mutex,
	uint32_t        state)
{
	/* release the interlock and re-enable preemption */
	lck_mtx_ilk_unlock_inline(mutex, state);
	LCK_MTX_TRY_ACQUIRED(mutex, mutex->lck_mtx_grp, false,
	    state & LCK_MTX_PROFILE_MSK);
}

__attribute__((always_inline))
static inline void
lck_mtx_convert_spin_finish_inline(
	lck_mtx_t       *mutex,
	uint32_t        state)
{
	/* release the interlock and acquire it as mutex */
	state &= ~(LCK_MTX_ILOCKED_MSK | LCK_MTX_SPIN_MSK);
	state |= LCK_MTX_MLOCKED_MSK;

	ordered_store_mtx_state_release(mutex, state);
	enable_preemption();
}

__attribute__((always_inline))
static inline void
lck_mtx_unlock_finish_inline(
	lck_mtx_t       *mutex,
	uint32_t        state)
{
	enable_preemption();
	LCK_MTX_RELEASED(mutex, mutex->lck_mtx_grp,
	    state & LCK_MTX_PROFILE_MSK);
}

#endif /* LCK_MTX_USE_ARCH */
#endif /* _I386_LOCKS_I386_INLINES_H_ */
