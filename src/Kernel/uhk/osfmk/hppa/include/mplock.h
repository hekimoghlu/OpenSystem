/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 5, 2024.
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
 * Copyright (c) 2004 Niklas Hallqvist.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 * IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
 * NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
 * THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef _MACHINE_MPLOCK_H_
#define _MACHINE_MPLOCK_H_

/*
 * Really simple spinlock implementation with recursive capabilities.
 * Correctness is paramount, no fanciness allowed.
 */

#define	MPL_LOCKED	0
#define	MPL_UNLOCKED	1

struct __mp_lock {
	volatile int mpl_lock[4];
	volatile struct cpu_info *mpl_cpu;
	volatile long mpl_count;
#ifdef WITNESS
	struct lock_object	mpl_lock_obj;
#endif
};

#ifndef _LOCORE

void	___mp_lock_init(struct __mp_lock *);
void	__mp_lock(struct __mp_lock *);
void	__mp_unlock(struct __mp_lock *);
int	__mp_release_all(struct __mp_lock *);
void	__mp_acquire_count(struct __mp_lock *, int);
int	__mp_lock_held(struct __mp_lock *, struct cpu_info *);

#ifdef WITNESS

void	_mp_lock_init(struct __mp_lock *, const struct lock_type *);

#define __mp_lock_init(mpl) do {					\
	static const struct lock_type __lock_type = { .lt_name = #mpl };\
	_mp_lock_init((mpl), &__lock_type);				\
} while (0)

#else /* WITNESS */

#define __mp_lock_init		___mp_lock_init

#endif /* WITNESS */

#endif

#endif /* !_MACHINE_MPLOCK_H */
