/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 13, 2022.
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
#ifndef MSCVER_STDATOMIC_H_
#define MSCVER_STDATOMIC_H_

#if !defined(__cplusplus) && defined(_MSC_VER)

#pragma warning(push)
#pragma warning(disable:4067)    /* newline for __has_include_next */

#if defined(__clang__) && __has_include_next(<stdatomic.h>)
   /* use the clang stdatomic.h with clang-cl*/
#  include_next <stdatomic.h>
#else /* ! stdatomic.h */

#include <windows.h>

#include "common/attributes.h"

typedef volatile LONG  atomic_int;
typedef volatile ULONG atomic_uint;

typedef enum {
    memory_order_relaxed,
    memory_order_acquire
} msvc_atomic_memory_order;

#define atomic_init(p_a, v)           do { *(p_a) = (v); } while(0)
#define atomic_store(p_a, v)          InterlockedExchange((LONG*)p_a, v)
#define atomic_load(p_a)              InterlockedCompareExchange((LONG*)p_a, 0, 0)
#define atomic_exchange(p_a, v)       InterlockedExchange(p_a, v)
#define atomic_load_explicit(p_a, mo) atomic_load(p_a)

/*
 * TODO use a special call to increment/decrement
 * using InterlockedIncrement/InterlockedDecrement
 */
#define atomic_fetch_add(p_a, inc)    InterlockedExchangeAdd(p_a, inc)
#define atomic_fetch_sub(p_a, dec)    InterlockedExchangeAdd(p_a, -(dec))
#define atomic_fetch_or(p_a, v)       InterlockedOr(p_a, v)

#endif /* ! stdatomic.h */

#pragma warning(pop)

#endif /* !defined(__cplusplus) && defined(_MSC_VER) */

#endif /* MSCVER_STDATOMIC_H_ */
