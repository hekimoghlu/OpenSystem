/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 14, 2025.
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
#ifndef GCCVER_STDATOMIC_H_
#define GCCVER_STDATOMIC_H_

#if !defined(__cplusplus)

typedef int atomic_int;
typedef unsigned int atomic_uint;

#define memory_order_relaxed __ATOMIC_RELAXED
#define memory_order_acquire __ATOMIC_ACQUIRE

#define atomic_init(p_a, v)           do { *(p_a) = (v); } while(0)
#define atomic_store(p_a, v)          __atomic_store_n(p_a, v, __ATOMIC_SEQ_CST)
#define atomic_load(p_a)              __atomic_load_n(p_a, __ATOMIC_SEQ_CST)
#define atomic_load_explicit(p_a, mo) __atomic_load_n(p_a, mo)
#define atomic_fetch_add(p_a, inc)    __atomic_fetch_add(p_a, inc, __ATOMIC_SEQ_CST)
#define atomic_fetch_sub(p_a, dec)    __atomic_fetch_sub(p_a, dec, __ATOMIC_SEQ_CST)
#define atomic_exchange(p_a, v)       __atomic_exchange_n(p_a, v, __ATOMIC_SEQ_CST)
#define atomic_fetch_or(p_a, v)       __atomic_fetch_or(p_a, v, __ATOMIC_SEQ_CST)

#endif /* !defined(__cplusplus) */

#endif /* GCCVER_STDATOMIC_H_ */
