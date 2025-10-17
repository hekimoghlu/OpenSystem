/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 25, 2023.
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
#include "VMHeap.h"
#include <thread>

namespace bmalloc {

VMHeap::VMHeap(std::lock_guard<Mutex>&)
{
}

LargeRange VMHeap::tryAllocateLargeChunk(size_t alignment, size_t size)
{
    // We allocate VM in aligned multiples to increase the chances that
    // the OS will provide contiguous ranges that we can merge.
    size_t roundedAlignment = roundUpToMultipleOf<chunkSize>(alignment);
    if (roundedAlignment < alignment) // Check for overflow
        return LargeRange();
    alignment = roundedAlignment;

    size_t roundedSize = roundUpToMultipleOf<chunkSize>(size);
    if (roundedSize < size) // Check for overflow
        return LargeRange();
    size = roundedSize;

    void* memory = tryVMAllocate(alignment, size);
    if (!memory)
        return LargeRange();
    
    Chunk* chunk = static_cast<Chunk*>(memory);
    
#if BOS(DARWIN)
    PerProcess<Zone>::get()->addRange(Range(chunk->bytes(), size));
#endif

    return LargeRange(chunk->bytes(), size, 0, 0);
}

} // namespace bmalloc
