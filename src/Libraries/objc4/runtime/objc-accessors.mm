/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 4, 2025.
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
#include <string.h>
#include <stddef.h>

#include "InitWrappers.h"
#include "objc-private.h"
#include "runtime.h"

// stub interface declarations to make compiler happy.

@interface __NSCopyable
- (id)copyWithZone:(void *)zone;
@end

@interface __NSMutableCopyable
- (id)mutableCopyWithZone:(void *)zone;
@end

objc::ExplicitInit<StripedMap<spinlock_t>> PropertyLocks;
objc::ExplicitInit<StripedMap<spinlock_t>> StructLocks;
objc::ExplicitInit<StripedMap<spinlock_t>> CppObjectLocks;

void accessors_init(void) {
    PropertyLocks.init();
    StructLocks.init();
    CppObjectLocks.init();
}

#define MUTABLE_COPY 2

id objc_getProperty(id self, SEL _cmd, ptrdiff_t offset, BOOL atomic) {
    if (offset == 0) {
        return object_getClass(self);
    }

    // Retain release world
    id *slot = (id*) ((char*)self + offset);
    if (!atomic) return *slot;
        
    // Atomic retain release world
    spinlock_t& slotlock = PropertyLocks.get()[slot];
    slotlock.lock();
    id value = objc_retain(*slot);
    slotlock.unlock();
    
    // for performance, we (safely) issue the autorelease OUTSIDE of the spinlock.
    return objc_autoreleaseReturnValue(value);
}


static inline void reallySetProperty(id self, SEL _cmd, id newValue, ptrdiff_t offset, bool atomic, bool copy, bool mutableCopy) __attribute__((always_inline));

static inline void reallySetProperty(id self, SEL _cmd, id newValue, ptrdiff_t offset, bool atomic, bool copy, bool mutableCopy)
{
    if (offset == 0) {
        object_setClass(self, newValue);
        return;
    }

    id oldValue;
    id *slot = (id*) ((char*)self + offset);

    if (copy) {
        newValue = [newValue copyWithZone:nil];
    } else if (mutableCopy) {
        newValue = [newValue mutableCopyWithZone:nil];
    } else {
        if (*slot == newValue) return;
        newValue = objc_retain(newValue);
    }

    if (!atomic) {
        oldValue = *slot;
        *slot = newValue;
    } else {
        spinlock_t& slotlock = PropertyLocks.get()[slot];
        slotlock.lock();
        oldValue = *slot;
        *slot = newValue;        
        slotlock.unlock();
    }

    objc_release(oldValue);
}

void objc_setProperty(id self, SEL _cmd, ptrdiff_t offset, id newValue, BOOL atomic, signed char shouldCopy) 
{
    bool copy = (shouldCopy && shouldCopy != MUTABLE_COPY);
    bool mutableCopy = (shouldCopy == MUTABLE_COPY);
    reallySetProperty(self, _cmd, newValue, offset, atomic, copy, mutableCopy);
}

void objc_setProperty_atomic(id self, SEL _cmd, id newValue, ptrdiff_t offset)
{
    reallySetProperty(self, _cmd, newValue, offset, true, false, false);
}

void objc_setProperty_nonatomic(id self, SEL _cmd, id newValue, ptrdiff_t offset)
{
    reallySetProperty(self, _cmd, newValue, offset, false, false, false);
}


void objc_setProperty_atomic_copy(id self, SEL _cmd, id newValue, ptrdiff_t offset)
{
    reallySetProperty(self, _cmd, newValue, offset, true, true, false);
}

void objc_setProperty_nonatomic_copy(id self, SEL _cmd, id newValue, ptrdiff_t offset)
{
    reallySetProperty(self, _cmd, newValue, offset, false, true, false);
}


// This entry point was designed wrong.  When used as a getter, src needs to be locked so that
// if simultaneously used for a setter then there would be contention on src.
// So we need two locks - one of which will be contended.
void objc_copyStruct(void *dest, const void *src, ptrdiff_t size, BOOL atomic, BOOL hasStrong __unused) {
    spinlock_t *srcLock = nil;
    spinlock_t *dstLock = nil;
    if (atomic) {
        srcLock = &StructLocks.get()[src];
        dstLock = &StructLocks.get()[dest];
        spinlock_t::lockTwo(srcLock, dstLock);
    }

    memmove(dest, src, size);

    if (atomic) {
        spinlock_t::unlockTwo(srcLock, dstLock);
    }
}

void objc_copyCppObjectAtomic(void *dest, const void *src, void (*copyHelper) (void *dest, const void *source)) {
    spinlock_t *srcLock = &CppObjectLocks.get()[src];
    spinlock_t *dstLock = &CppObjectLocks.get()[dest];
    spinlock_t::lockTwo(srcLock, dstLock);

    // let C++ code perform the actual copy.
    copyHelper(dest, src);
    
    spinlock_t::unlockTwo(srcLock, dstLock);
}
