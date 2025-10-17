/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 9, 2023.
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
#ifndef __LOCKING_H
#define __LOCKING_H

#if OS_UNFAIR_LOCK_INLINE
#define os_unfair_lock_lock_with_options(lock, options) \
		os_unfair_lock_lock_with_options_inline(lock, options)
#define os_unfair_lock_trylock(lock) \
		os_unfair_lock_trylock_inline(lock)
#define os_unfair_lock_unlock(lock) \
		os_unfair_lock_unlock_inline(lock)
#endif // OS_UNFAIR_LOCK_INLINE

typedef os_unfair_lock _malloc_lock_s;
#define _MALLOC_LOCK_INIT OS_UNFAIR_LOCK_INIT

__attribute__((always_inline))
static inline void
_malloc_lock_init(_malloc_lock_s *lock) {
    *lock = OS_UNFAIR_LOCK_INIT;
}

MALLOC_ALWAYS_INLINE
static inline void
_malloc_lock_lock(_malloc_lock_s *lock) {
	return os_unfair_lock_lock_with_options(lock, OS_UNFAIR_LOCK_ADAPTIVE_SPIN |
			OS_UNFAIR_LOCK_DATA_SYNCHRONIZATION);
}

MALLOC_ALWAYS_INLINE
static inline bool
_malloc_lock_trylock(_malloc_lock_s *lock) {
    return os_unfair_lock_trylock(lock);
}

MALLOC_ALWAYS_INLINE
static inline void
_malloc_lock_unlock(_malloc_lock_s *lock) {
    return os_unfair_lock_unlock(lock);
}

MALLOC_ALWAYS_INLINE
static inline void
_malloc_lock_assert_owner(_malloc_lock_s *lock) {
	os_unfair_lock_assert_owner(lock);
}

#endif // __LOCKING_H
