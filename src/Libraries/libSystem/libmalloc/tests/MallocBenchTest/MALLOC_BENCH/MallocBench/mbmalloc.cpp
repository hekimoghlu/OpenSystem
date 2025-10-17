/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 16, 2024.
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
#include <limits>
#include <stdio.h>
#include <stdlib.h>

#ifdef __APPLE__
#import <malloc/malloc.h>
#else
#include <malloc.h>
#endif

extern "C" {

void* mbmalloc(size_t size)
{
    return malloc(size);
}

void* mbmemalign(size_t alignment, size_t size)
{
    void* result;
    if (posix_memalign(&result, alignment, size))
        return nullptr;
    return result;
}

void mbfree(void* p, size_t)
{
    return free(p);
}

void* mbrealloc(void* p, size_t, size_t newSize)
{
    return realloc(p, newSize);
}

void mbscavenge()
{
#ifdef __APPLE__
    malloc_zone_pressure_relief(nullptr, 0);
#else
    malloc_trim(0);
#endif
}

} // extern "C"
