/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 26, 2024.
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
#include "config.h"

#ifdef HAVE_SYS_TYPES_H
#include <sys/types.h>
#endif
#ifdef HAVE_SYS_SELECT_H
#include <sys/select.h>
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <string.h>
#include <limits.h>

#ifdef HAVE_UNISTD_H
#include <unistd.h>
#endif

#include "heimqueue.h"
#include "heim_threads.h"
#include "heimbase.h"
#include "heimbasepriv.h"
#include "heimbase_impl.h"

#ifdef HAVE_DISPATCH_DISPATCH_H
#include <dispatch/dispatch.h>
#endif

#if defined(__GNUC__) && defined(HAVE___SYNC_ADD_AND_FETCH)

#define heim_base_atomic_inc(x) __sync_add_and_fetch((x), 1)
#define heim_base_atomic_dec(x) __sync_sub_and_fetch((x), 1)
#define heim_base_atomic_type	unsigned int
#define heim_base_atomic_max    UINT_MAX

#ifndef __has_builtin
#define __has_builtin(x) 0
#endif

#if __has_builtin(__sync_swap)
#define heim_base_exchange_pointer(t,v) __sync_swap((t), (v))
#else
#define heim_base_exchange_pointer(t,v) __sync_lock_test_and_set((t), (v))
#endif

#elif defined(_WIN32)

#define heim_base_atomic_inc(x) InterlockedIncrement(x)
#define heim_base_atomic_dec(x) InterlockedDecrement(x)
#define heim_base_atomic_type	LONG
#define heim_base_atomic_max    MAXLONG

#define heim_base_exchange_pointer(t,v) InterlockedExchangePointer((t),(v))

#else

#define HEIM_BASE_NEED_ATOMIC_MUTEX 1
extern HEIMDAL_MUTEX _heim_base_mutex;

#define heim_base_atomic_type	unsigned int

static inline heim_base_atomic_type
heim_base_atomic_inc(heim_base_atomic_type *x)
{
    heim_base_atomic_type t;
    HEIMDAL_MUTEX_lock(&_heim_base_mutex);
    t = ++(*x);
    HEIMDAL_MUTEX_unlock(&_heim_base_mutex);
    return t;
}

static inline heim_base_atomic_type
heim_base_atomic_dec(heim_base_atomic_type *x)
{
    heim_base_atomic_type t;
    HEIMDAL_MUTEX_lock(&_heim_base_mutex);
    t = --(*x);
    HEIMDAL_MUTEX_unlock(&_heim_base_mutex);
    return t;
}

#define heim_base_atomic_max    UINT_MAX

#endif

/* tagged strings/object/XXX */
#define heim_base_is_tagged(x) (((uintptr_t)(x)) & 0x3)

#define heim_base_is_tagged_object(x) ((((uintptr_t)(x)) & 0x3) == 1)
#define heim_base_make_tagged_object(x, tid) \
    ((heim_object_t)((((uintptr_t)(x)) << 5) | ((tid) << 2) | 0x1))
#define heim_base_tagged_object_tid(x) ((((uintptr_t)(x)) & 0x1f) >> 2)
#define heim_base_tagged_object_value(x) (((uintptr_t)(x)) >> 5)

/*
 *
 */

#undef HEIMDAL_NORETURN_ATTRIBUTE
#define HEIMDAL_NORETURN_ATTRIBUTE
#undef HEIMDAL_PRINTF_ATTRIBUTE
#define HEIMDAL_PRINTF_ATTRIBUTE(x)
