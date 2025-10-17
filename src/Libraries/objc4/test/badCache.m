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
#include "test.h"

// Test objc_msgSend's detection of infinite loops during cache scan.

#if __arm__

int main()
{
    testwarn("objc_msgSend on arm doesn't detect infinite loops");
    fprintf(stderr, "arm\n");
    succeed(__FILE__);
}

#else

#include "testroot.i"

#if __LP64__
typedef uint32_t mask_t;
#else
typedef uint16_t mask_t;
#endif

struct bucket_t {
    uintptr_t sel;
    uintptr_t imp;
};

struct cache_t {
    uintptr_t buckets;
};

struct cache_outlined_or_64 {
    struct cache_t cache;
    mask_t mask;
    mask_t occupied;
};

struct cache_inline_32 {
    struct cache_t cache;
    mask_t occupied;
};

struct class_t {
    void *isa;
    void *supercls;
    struct cache_t cache;
};

@interface Subclass : TestRoot @end
@implementation Subclass @end

int main()
{
    Class cls = [TestRoot class];
    id obj = [cls new];
    [obj self];

    struct cache_t *cache = &((__bridge struct class_t *)cls)->cache;
    
    // Figure out which cache mask scheme is in use by examining the existing bits.
    int low4 = 0;
#if __LP64__
    int top16 = 0;
#endif
    int outlined = 0;
  
    if (cache->buckets & 0xf) {
        low4 = 1;
#if __LP64__
    } else if ((cache->buckets & (0xffffULL << 48))) {
        top16 = 1;
#endif
    } else {
        outlined = 1;
    }
    
#   define COUNT 4
#   define COUNTSHIFT 14
    struct bucket_t *buckets = (struct bucket_t *)calloc(sizeof(struct bucket_t), COUNT+1);
    for (int i = 0; i < COUNT; i++) {
        buckets[i].sel = ~0;
        buckets[i].imp = ~0;
    }
    buckets[COUNT].sel = 1;
    buckets[COUNT].imp = (uintptr_t)buckets;

    if (low4) {
        cache->buckets = (uintptr_t)buckets | COUNTSHIFT;
        ((struct cache_inline_32 *)cache)->occupied = 0;
#if __LP64__
    } else if (top16) {
        cache->buckets = ((uintptr_t)(COUNT - 1) << 48) | (uintptr_t)buckets;
        ((struct cache_outlined_or_64 *)cache)->occupied = 0;
#endif
    } else if (outlined) {
        ((struct cache_outlined_or_64 *)cache)->mask = COUNT-1;
        cache->buckets = (uintptr_t)buckets;
        ((struct cache_outlined_or_64 *)cache)->occupied = 0;
    }


    fprintf(stderr, "crash now\n");
    [obj self];

    fail("should have crashed");
}

#endif
