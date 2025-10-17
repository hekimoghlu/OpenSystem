/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 5, 2024.
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

#if ENABLE_PHYSICAL_PAGE_MAP 

#include "VMAllocate.h"
#include <unordered_set>

#if !BUSE(LIBPAS)

namespace bmalloc {

// This class is useful for debugging bmalloc's footprint.
class PhysicalPageMap {
public:

    void commit(void* ptr, size_t size)
    {
        forEachPhysicalPage(ptr, size, [&] (void* ptr) {
            m_physicalPages.insert(ptr);
        });
    }

    void decommit(void* ptr, size_t size)
    {
        forEachPhysicalPage(ptr, size, [&] (void* ptr) {
            m_physicalPages.erase(ptr);
        });
    }

    size_t footprint()
    {
        return static_cast<size_t>(m_physicalPages.size()) * vmPageSizePhysical();
    }

private:
    template <typename F>
    void forEachPhysicalPage(void* ptr, size_t size, F f)
    {
        char* begin = roundUpToMultipleOf(vmPageSizePhysical(), static_cast<char*>(ptr));
        char* end = roundDownToMultipleOf(vmPageSizePhysical(), static_cast<char*>(ptr) + size);
        while (begin < end) {
            f(begin);
            begin += vmPageSizePhysical();
        }
    }

    std::unordered_set<void*> m_physicalPages;
};

} // namespace bmalloc

#endif
#endif // ENABLE_PHYSICAL_PAGE_MAP 
