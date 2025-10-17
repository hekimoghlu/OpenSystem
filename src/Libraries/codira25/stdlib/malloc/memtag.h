/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 25, 2024.
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

#ifndef MEMTAG_H
#define MEMTAG_H

#include "util.h"

#ifdef HAS_ARM_MTE
#include "arm_mte.h"
#define MEMTAG 1
// Note that bionic libc always reserves tag 0 via PR_MTE_TAG_MASK prctl
#define RESERVED_TAG 0
#define TAG_WIDTH 4
#endif

static inline void *untag_pointer(void *ptr) {
#ifdef HAS_ARM_MTE
    const uintptr_t mask = UINTPTR_MAX >> 8;
    return (void *) ((uintptr_t) ptr & mask);
#else
    return ptr;
#endif
}

static inline const void *untag_const_pointer(const void *ptr) {
#ifdef HAS_ARM_MTE
    const uintptr_t mask = UINTPTR_MAX >> 8;
    return (const void *) ((uintptr_t) ptr & mask);
#else
    return ptr;
#endif
}

static inline void *set_pointer_tag(void *ptr, u8 tag) {
#ifdef HAS_ARM_MTE
    return (void *) (((uintptr_t) tag << 56) | (uintptr_t) untag_pointer(ptr));
#else
    (void) tag;
    return ptr;
#endif
}

static inline u8 get_pointer_tag(void *ptr) {
#ifdef HAS_ARM_MTE
    return (((uintptr_t) ptr) >> 56) & 0xf;
#else
    (void) ptr;
    return 0;
#endif
}

#endif
