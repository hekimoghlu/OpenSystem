/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 28, 2022.
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
#include "PerProcess.h"

#include "VMAllocate.h"
#include <stdio.h>

#if !BUSE(LIBPAS)

namespace bmalloc {

static constexpr unsigned tableSize = 100;
static constexpr bool verbose = false;

static Mutex s_mutex;

static char* s_bumpBase;
static size_t s_bumpOffset;
static size_t s_bumpLimit;

static PerProcessData* s_table[tableSize];

// Returns zero-filled memory by virtual of using vmAllocate.
static void* allocate(size_t size, size_t alignment)
{
    s_bumpOffset = roundUpToMultipleOf(alignment, s_bumpOffset);
    void* result = s_bumpBase + s_bumpOffset;
    s_bumpOffset += size;
    if (s_bumpOffset <= s_bumpLimit)
        return result;
    
    size_t allocationSize = vmSize(size);
    void* allocation = vmAllocate(allocationSize);
    s_bumpBase = static_cast<char*>(allocation);
    s_bumpOffset = 0;
    s_bumpLimit = allocationSize;
    return allocate(size, alignment);
}

PerProcessData* getPerProcessData(unsigned hash, const char* disambiguator, size_t size, size_t alignment)
{
    LockHolder lock(s_mutex);

    PerProcessData*& bucket = s_table[hash % tableSize];
    
    for (PerProcessData* data = bucket; data; data = data->next) {
        if (!strcmp(data->disambiguator, disambiguator)) {
            if (verbose)
                fprintf(stderr, "%d: Using existing %s\n", getpid(), disambiguator);
            RELEASE_BASSERT(data->size == size);
            RELEASE_BASSERT(data->alignment == alignment);
            return data;
        }
    }
    
    if (verbose)
        fprintf(stderr, "%d: Creating new %s\n", getpid(), disambiguator);
    void* rawDataPtr = allocate(sizeof(PerProcessData), std::alignment_of<PerProcessData>::value);
    PerProcessData* data = static_cast<PerProcessData*>(rawDataPtr);
    data->disambiguator = disambiguator;
    data->memory = allocate(size, alignment);
    data->size = size;
    data->alignment = alignment;
    data->next = bucket;
    bucket = data;
    return data;
}

} // namespace bmalloc

#endif
