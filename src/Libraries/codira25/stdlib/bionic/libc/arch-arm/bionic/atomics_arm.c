/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 9, 2024.
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
/* The purpose of this file is to export a small set of atomic-related
 * functions from the C library, to ensure binary ABI compatibility for
 * the NDK.
 *
 * These functions were initially exposed by the NDK through <sys/atomics.h>,
 * which was unfortunate because their implementation didn't provide any
 * memory barriers at all.
 *
 * This wasn't a problem for the platform code that used them, because it
 * used explicit barrier instructions around them. On the other hand, it means
 * that any NDK-generated machine code that linked against them would not
 * perform correctly when running on multi-core devices.
 *
 * To fix this, the platform code was first modified to not use any of these
 * functions (everything is now inlined through assembly statements, see
 * libc/private/bionic_arm_inline.h and the headers it includes.
 *
 * The functions here are thus only for the benefit of NDK applications,
 * and now includes full memory barriers to prevent any random memory ordering
 * issue from cropping.
 *
 * Note that we also provide an updated <sys/atomics.h> header that defines
 * always_inlined versions of the functions that use the GCC builtin
 * intrinsics to perform the same thing.
 *
 * NOTE: There is no need for a similar file for non-ARM platforms.
 */

/* DO NOT INCLUDE <sys/atomics.h> HERE ! */

int
__atomic_cmpxchg(int old, int _new, volatile int *ptr)
{
    /* We must return 0 on success */
    return __sync_val_compare_and_swap(ptr, old, _new) != old;
}

int
__atomic_swap(int _new, volatile int *ptr)
{
    int prev;
    do {
        prev = *ptr;
    } while (__sync_val_compare_and_swap(ptr, prev, _new) != prev);
    return prev;
}

int
__atomic_dec(volatile int *ptr)
{
  return __sync_fetch_and_sub (ptr, 1);
}

int
__atomic_inc(volatile int *ptr)
{
  return __sync_fetch_and_add (ptr, 1);
}
