/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 26, 2024.
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
/**
 * @file
 * @brief
 */

#ifndef ZYCORE_ALLOCATOR_H
#define ZYCORE_ALLOCATOR_H

#include "ZycoreExportConfig.h"
#include "ZycoreStatus.h"
#include "ZycoreTypes.h"
#include <wtf/Compiler.h>

WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================================== */
/* Enums and types                                                                                */
/* ============================================================================================== */

struct ZyanAllocator_;

/**
 * Defines the `ZyanAllocatorAllocate` function prototype.
 *
 * @param   allocator       A pointer to the `ZyanAllocator` instance.
 * @param   p               Receives a pointer to the first memory block sufficient to hold an
 *                          array of `n` elements with a size of `element_size`.
 * @param   element_size    The size of a single element.
 * @param   n               The number of elements to allocate storage for.
 *
 * @return  A zyan status code.
 *
 * This prototype is used for the `allocate()` and `reallocate()` functions.
 *
 * The result of the `reallocate()` function is undefined, if `p` does not point to a memory block
 * previously obtained by `(re-)allocate()`.
 */
typedef ZyanStatus (*ZyanAllocatorAllocate)(struct ZyanAllocator_* allocator, void** p,
    ZyanUSize element_size, ZyanUSize n);

/**
 * Defines the `ZyanAllocatorDeallocate` function prototype.
 *
 * @param   allocator       A pointer to the `ZyanAllocator` instance.
 * @param   p               The pointer obtained from `(re-)allocate()`.
 * @param   element_size    The size of a single element.
 * @param   n               The number of elements earlier passed to `(re-)allocate()`.
 *
  * @return  A zyan status code.
 */
typedef ZyanStatus (*ZyanAllocatorDeallocate)(struct ZyanAllocator_* allocator, void* p,
    ZyanUSize element_size, ZyanUSize n);

/**
 * Defines the `ZyanAllocator` struct.
 *
 * This is the base class for all custom allocator implementations.
 *
 * All fields in this struct should be considered as "private". Any changes may lead to unexpected
 * behavior.
 */
typedef struct ZyanAllocator_
{
    /**
     * The allocate function.
     */
    ZyanAllocatorAllocate allocate;
    /**
     * The reallocate function.
     */
    ZyanAllocatorAllocate reallocate;
    /**
     * The deallocate function.
     */
    ZyanAllocatorDeallocate deallocate;
} ZyanAllocator;

/* ============================================================================================== */
/* Exported functions                                                                             */
/* ============================================================================================== */

/**
 * Initializes the given `ZyanAllocator` instance.
 *
 * @param   allocator   A pointer to the `ZyanAllocator` instance.
 * @param   allocate    The allocate function.
 * @param   reallocate  The reallocate function.
 * @param   deallocate  The deallocate function.
 *
 * @return  A zyan status code.
 */
ZYCORE_EXPORT ZyanStatus ZyanAllocatorInit(ZyanAllocator* allocator, ZyanAllocatorAllocate allocate,
    ZyanAllocatorAllocate reallocate, ZyanAllocatorDeallocate deallocate);

#ifndef ZYAN_NO_LIBC

/**
 * Returns the default `ZyanAllocator` instance.
 *
 * @return  A pointer to the default `ZyanAllocator` instance.
 *
 * The default allocator uses the default memory manager to allocate memory on the heap.
 *
 * You should in no case modify the returned allocator instance to avoid unexpected behavior.
 */
ZYCORE_EXPORT ZYAN_REQUIRES_LIBC ZyanAllocator* ZyanAllocatorDefault(void);

#endif // ZYAN_NO_LIBC

/* ============================================================================================== */

#ifdef __cplusplus
}
#endif

WTF_ALLOW_UNSAFE_BUFFER_USAGE_END

#endif /* ZYCORE_ALLOCATOR_H */
