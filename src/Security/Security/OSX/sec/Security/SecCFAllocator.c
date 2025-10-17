/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 7, 2022.
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
#include <Security/SecCFAllocator.h>
#include <CoreFoundation/CoreFoundation.h>
#include <corecrypto/cc.h>
#include <malloc/malloc.h>

static CFAllocatorContext sDefaultCtx;

static CFStringRef SecCFAllocatorCopyDescription(const void *info) {
    return CFSTR("Custom CFAllocator for sensitive data that zeroizes on deallocate");
}

// primary goal of this allocator is to clear memory when it is deallocated
static void SecCFAllocatorDeallocate(void *ptr, void *info) {
    if (!ptr) return;
    size_t sz = malloc_size(ptr);
    if(sz) cc_clear(sz, ptr);

    sDefaultCtx.deallocate(ptr, info);
}

CFAllocatorRef SecCFAllocatorZeroize(void) {
    static dispatch_once_t sOnce = 0;
    static CFAllocatorRef sAllocator = NULL;
    dispatch_once(&sOnce, ^{
        CFAllocatorGetContext(kCFAllocatorMallocZone, &sDefaultCtx);

        CFAllocatorContext ctx = {0,
            sDefaultCtx.info,
            sDefaultCtx.retain,
            sDefaultCtx.release,
            SecCFAllocatorCopyDescription,
            sDefaultCtx.allocate,
            sDefaultCtx.reallocate,
            SecCFAllocatorDeallocate,
            sDefaultCtx.preferredSize};

        sAllocator = CFAllocatorCreate(NULL, &ctx);
    });

    return sAllocator;
}
