/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 19, 2021.
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

#include "VMAllocate.h"
#include <vector>

namespace bmalloc {

class BulkDecommit {
    using Data = std::vector<std::pair<char*, size_t>>;

public:
    void addEager(void* ptr, size_t size)
    {
        add(m_eager, ptr, size);
    }
    void addLazy(void* ptr, size_t size)
    {
        add(m_lazy, ptr, size);
    }
    void processEager()
    {
        process(m_eager);
    }
    void processLazy()
    {
        process(m_lazy);
    }

private:
    void add(Data& data, void* ptr, size_t size)
    {
        char* begin = roundUpToMultipleOf(vmPageSizePhysical(), static_cast<char*>(ptr));
        char* end = roundDownToMultipleOf(vmPageSizePhysical(), static_cast<char*>(ptr) + size);
        if (begin >= end)
            return;
        data.push_back({begin, end - begin});
    }

    void process(BulkDecommit::Data& decommits)
    {
        std::sort(
            decommits.begin(), decommits.end(),
            [&] (const auto& a, const auto& b) -> bool {
                return a.first < b.first;
            });

        char* run = nullptr;
        size_t runSize = 0;
        for (unsigned i = 0; i < decommits.size(); ++i) {
            auto& pair = decommits[i];
            if (run + runSize != pair.first) {
                if (run)
                    vmDeallocatePhysicalPages(run, runSize);
                run = pair.first;
                runSize = pair.second;
            } else {
                BASSERT(run);
                runSize += pair.second;
            }
        }

        if (run)
            vmDeallocatePhysicalPages(run, runSize);
    }

    Data m_eager;
    Data m_lazy;
};

} // namespace bmalloc
