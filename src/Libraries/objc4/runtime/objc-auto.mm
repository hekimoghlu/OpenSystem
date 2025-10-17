/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 6, 2024.
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
#define OBJC_DECLARE_SYMBOLS 1
#include "objc-private.h"
#include "objc-auto.h"

// GC is no longer supported.

#if OBJC_NO_GC_API

// No GC and no GC symbols needed. We're done here.
# if SUPPORT_GC_COMPAT
#   error inconsistent config settings
# endif

#else

// No GC but we do need to export GC symbols.

# if !SUPPORT_GC_COMPAT
#   error inconsistent config settings
# endif

void objc_collect(unsigned long options __unused) { }
BOOL objc_collectingEnabled(void) { return NO; }
void objc_setCollectionThreshold(size_t threshold __unused) { }
void objc_setCollectionRatio(size_t ratio __unused) { }
void objc_startCollectorThread(void) { }

BOOL objc_atomicCompareAndSwapPtr(id predicate, id replacement, volatile id *objectLocation) 
    { return CompareAndSwapNoBarrier(predicate, replacement, objectLocation); }

BOOL objc_atomicCompareAndSwapPtrBarrier(id predicate, id replacement, volatile id *objectLocation) 
    { return CompareAndSwap(predicate, replacement, objectLocation); }

BOOL objc_atomicCompareAndSwapGlobal(id predicate, id replacement, volatile id *objectLocation) 
    { return objc_atomicCompareAndSwapPtr(predicate, replacement, objectLocation); }

BOOL objc_atomicCompareAndSwapGlobalBarrier(id predicate, id replacement, volatile id *objectLocation) 
    { return objc_atomicCompareAndSwapPtrBarrier(predicate, replacement, objectLocation); }

BOOL objc_atomicCompareAndSwapInstanceVariable(id predicate, id replacement, volatile id *objectLocation) 
    { return objc_atomicCompareAndSwapPtr(predicate, replacement, objectLocation); }

BOOL objc_atomicCompareAndSwapInstanceVariableBarrier(id predicate, id replacement, volatile id *objectLocation) 
    { return objc_atomicCompareAndSwapPtrBarrier(predicate, replacement, objectLocation); }

id objc_assign_strongCast(id val, id *dest) 
    { return (*dest = val); }

id objc_assign_global(id val, id *dest) 
    { return (*dest = val); }

id objc_assign_threadlocal(id val, id *dest)
    { return (*dest = val); }

id objc_assign_ivar(id val, id dest, ptrdiff_t offset) 
    { return (*(id*)((char *)dest+offset) = val); }

id objc_read_weak(id *location) 
    { return *location; }

id objc_assign_weak(id value, id *location) 
    { return (*location = value); }

void *objc_memmove_collectable(void *dst, const void *src, size_t size) 
    { return memmove(dst, src, size); }

void objc_finalizeOnMainThread(Class cls __unused) { }
BOOL objc_is_finalized(void *ptr __unused) { return NO; }
void objc_clear_stack(unsigned long options __unused) { }

BOOL objc_collecting_enabled(void) { return NO; }
void objc_set_collection_threshold(size_t threshold __unused) { } 
void objc_set_collection_ratio(size_t ratio __unused) { } 
void objc_start_collector_thread(void) { }

id objc_allocate_object(Class cls, int extra) 
    { return class_createInstance(cls, extra); }

void objc_registerThreadWithCollector() { }
void objc_unregisterThreadWithCollector() { }
void objc_assertRegisteredThreadWithCollector() { }

malloc_zone_t* objc_collect_init(int(*callback)() __unused) { return nil; }
malloc_zone_t* objc_collectableZone() { return nil; }

BOOL objc_isAuto(id object __unused) { return NO; }
BOOL objc_dumpHeap(char *filename __unused, unsigned long length __unused)
    { return NO; }

// not OBJC_NO_GC_API
#endif
