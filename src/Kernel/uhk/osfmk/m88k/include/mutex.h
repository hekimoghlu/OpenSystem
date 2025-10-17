/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 25, 2024.
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
 * Copyright (c) 2005, Miodrag Vallat.
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
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT,
 * INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 * STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 * ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef _M88K_MUTEX_H_
#define _M88K_MUTEX_H_

#include <sys/_lock.h>

struct mutex {
	volatile int mtx_lock;
	int mtx_wantipl;
	int mtx_oldipl;
	volatile void *mtx_owner;
#ifdef WITNESS
	struct lock_object mtx_lock_obj;
#endif
};

#ifdef WITNESS
#define	MUTEX_INITIALIZER_FLAGS(ipl, name, flags) \
	{ 0, __MUTEX_IPL((ipl)), IPL_NONE, NULL, \
	  MTX_LO_INITIALIZER(name, flags) }
#else
#define	MUTEX_INITIALIZER_FLAGS(ipl, name, flags) \
	{ 0, __MUTEX_IPL((ipl)), IPL_NONE, NULL }
#endif

void __mtx_init(struct mutex *, int);
#define _mtx_init(mtx, ipl) __mtx_init((mtx), __MUTEX_IPL((ipl)))

#ifdef DIAGNOSTIC

#define MUTEX_ASSERT_LOCKED(mtx) do {					\
	if (((mtx)->mtx_owner != curcpu()) && !(panicstr || db_active))	\
		panic("mutex %p not held in %s", (mtx), __func__);	\
} while (0)

#define MUTEX_ASSERT_UNLOCKED(mtx) do {					\
	if (((mtx)->mtx_owner == curcpu()) && !(panicstr || db_active))	\
		panic("mutex %p held in %s", (mtx), __func__);		\
} while (0)

#else

#define	MUTEX_ASSERT_LOCKED(mtx)	do { /* nothing */ } while (0)
#define	MUTEX_ASSERT_UNLOCKED(mtx)	do { /* nothing */ } while (0)

#endif

#define MUTEX_LOCK_OBJECT(mtx)	(&(mtx)->mtx_lock_obj)
#define MUTEX_OLDIPL(mtx)	(mtx)->mtx_oldipl

#endif	/* _M88K_MUTEX_H_ */
