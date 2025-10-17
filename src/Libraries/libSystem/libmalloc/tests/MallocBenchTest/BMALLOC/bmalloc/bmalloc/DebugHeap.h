/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 4, 2023.
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
#pragma once

#include "Mutex.h"
#include <mutex>
#include <unordered_map>

#if BOS(DARWIN)
#include <malloc/malloc.h>
#endif

namespace bmalloc {
    
class DebugHeap {
public:
    DebugHeap(std::lock_guard<Mutex>&);
    
    void* malloc(size_t);
    void* memalign(size_t alignment, size_t, bool crashOnFailure);
    void* realloc(void*, size_t, bool crashOnFailure);
    void free(void*);
    
    void* memalignLarge(size_t alignment, size_t);
    void freeLarge(void* base);

private:
#if BOS(DARWIN)
    malloc_zone_t* m_zone;
#endif
    
    // This is the debug heap. We can use whatever data structures we like. It doesn't matter.
    size_t m_pageSize { 0 };
    std::mutex m_lock;
    std::unordered_map<void*, size_t> m_sizeMap;
};

} // namespace bmalloc
