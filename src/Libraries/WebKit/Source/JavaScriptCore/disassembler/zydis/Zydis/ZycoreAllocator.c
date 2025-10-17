/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 11, 2024.
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

#if ENABLE(ZYDIS)

#include "ZycoreAllocator.h"
#include "ZycoreLibC.h"
#include <wtf/Compiler.h>

WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN

/* ============================================================================================== */
/* Internal functions                                                                             */
/* ============================================================================================== */

/* ---------------------------------------------------------------------------------------------- */
/* Default allocator                                                                              */
/* ---------------------------------------------------------------------------------------------- */

#ifndef ZYAN_NO_LIBC

static ZyanStatus ZyanAllocatorDefaultAllocate(ZyanAllocator* allocator, void** p,
    ZyanUSize element_size, ZyanUSize n)
{
    ZYAN_ASSERT(allocator);
    ZYAN_ASSERT(p);
    ZYAN_ASSERT(element_size);
    ZYAN_ASSERT(n);

    ZYAN_UNUSED(allocator);

    *p = ZYAN_MALLOC(element_size * n);
    if (!*p)
    {
        return ZYAN_STATUS_NOT_ENOUGH_MEMORY;
    }

    return ZYAN_STATUS_SUCCESS;
}

static ZyanStatus ZyanAllocatorDefaultReallocate(ZyanAllocator* allocator, void** p,
    ZyanUSize element_size, ZyanUSize n)
{
    ZYAN_ASSERT(allocator);
    ZYAN_ASSERT(p);
    ZYAN_ASSERT(element_size);
    ZYAN_ASSERT(n);

    ZYAN_UNUSED(allocator);

    void* const x = ZYAN_REALLOC(*p, element_size * n);
    if (!x)
    {
        return ZYAN_STATUS_NOT_ENOUGH_MEMORY;
    }
    *p = x;

    return ZYAN_STATUS_SUCCESS;
}

static ZyanStatus ZyanAllocatorDefaultDeallocate(ZyanAllocator* allocator, void* p,
    ZyanUSize element_size, ZyanUSize n)
{
    ZYAN_ASSERT(allocator);
    ZYAN_ASSERT(p);
    ZYAN_ASSERT(element_size);
    ZYAN_ASSERT(n);

    ZYAN_UNUSED(allocator);
    ZYAN_UNUSED(element_size);
    ZYAN_UNUSED(n);

    ZYAN_FREE(p);

    return ZYAN_STATUS_SUCCESS;
}

#endif // ZYAN_NO_LIBC

/* ---------------------------------------------------------------------------------------------- */

/* ============================================================================================== */
/* Exported functions                                                                             */
/* ============================================================================================== */

ZyanStatus ZyanAllocatorInit(ZyanAllocator* allocator, ZyanAllocatorAllocate allocate,
    ZyanAllocatorAllocate reallocate, ZyanAllocatorDeallocate deallocate)
{
    if (!allocator || !allocate || !reallocate || !deallocate)
    {
        return ZYAN_STATUS_INVALID_ARGUMENT;
    }

    allocator->allocate   = allocate;
    allocator->reallocate = reallocate;
    allocator->deallocate = deallocate;

    return ZYAN_STATUS_SUCCESS;
}

#ifndef ZYAN_NO_LIBC

ZyanAllocator* ZyanAllocatorDefault(void)
{
    static ZyanAllocator allocator =
    {
        &ZyanAllocatorDefaultAllocate,
        &ZyanAllocatorDefaultReallocate,
        &ZyanAllocatorDefaultDeallocate
    };
    return &allocator;
}

#endif

/* ============================================================================================== */

WTF_ALLOW_UNSAFE_BUFFER_USAGE_END

#endif /* ENABLE(ZYDIS) */
