/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 3, 2022.
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
#include <wtf/MemoryFootprint.h>

#include <algorithm>
#include <type_traits>
#include <windows.h>
#include <psapi.h>
#include <wtf/MallocSpan.h>
#include <wtf/win/Win32Handle.h>

namespace WTF {

size_t memoryFootprint()
{
    // We would like to calculate size of private working set.
    // https://msdn.microsoft.com/en-us/library/windows/desktop/ms684891(v=vs.85).aspx
    // > The working set of a program is a collection of those pages in its virtual address
    // > space that have been recently referenced. It includes both shared and private data.
    // > The shared data includes pages that contain all instructions your application executes,
    // > including those in your DLLs and the system DLLs. As the working set size increases,
    // > memory demand increases.
    auto process = Win32Handle::adopt(::OpenProcess(PROCESS_QUERY_INFORMATION, FALSE, GetCurrentProcessId()));
    if (!process)
        return 0;

    auto countSizeOfPrivateWorkingSet = [] (const PSAPI_WORKING_SET_INFORMATION& workingSets) {
        constexpr const size_t pageSize = 4 * KB;
        size_t numberOfPrivateWorkingSetPages = 0;
        for (size_t i = 0; i < workingSets.NumberOfEntries; ++i) {
            // https://msdn.microsoft.com/en-us/library/windows/desktop/ms684902(v=vs.85).aspx
            PSAPI_WORKING_SET_BLOCK workingSetBlock = workingSets.WorkingSetInfo[i];
            if (!workingSetBlock.Shared)
                numberOfPrivateWorkingSetPages++;
        }
        return numberOfPrivateWorkingSetPages * pageSize;
    };

    // https://msdn.microsoft.com/en-us/library/windows/desktop/ms684946(v=vs.85).aspx
    constexpr const size_t minNumberOfEntries = 16;
    constexpr const size_t sizeOfBufferOnStack = sizeof(PSAPI_WORKING_SET_INFORMATION) + minNumberOfEntries * sizeof(PSAPI_WORKING_SET_BLOCK);
    alignas(PSAPI_WORKING_SET_INFORMATION) std::byte bufferOnStack[sizeOfBufferOnStack];
    auto* workingSetsOnStack = reinterpret_cast<PSAPI_WORKING_SET_INFORMATION*>(&bufferOnStack);
    if (QueryWorkingSet(process.get(), workingSetsOnStack, sizeOfBufferOnStack))
        return countSizeOfPrivateWorkingSet(*workingSetsOnStack);

    auto updateNumberOfEntries = [&] (size_t numberOfEntries) {
        // If working set increases between first QueryWorkingSet and second QueryWorkingSet, the second one can fail.
        // At that time, we should increase numberOfEntries.
        return std::max(minNumberOfEntries, numberOfEntries + numberOfEntries / 4 + 1);
    };

    for (size_t numberOfEntries = updateNumberOfEntries(workingSetsOnStack->NumberOfEntries);;) {
        size_t workingSetSizeInBytes = sizeof(PSAPI_WORKING_SET_INFORMATION) + sizeof(PSAPI_WORKING_SET_BLOCK) * numberOfEntries;
        auto workingSets = MallocSpan<PSAPI_WORKING_SET_INFORMATION>::malloc(workingSetSizeInBytes);
        auto workingSetsSpan = workingSets.mutableSpan();
        if (QueryWorkingSet(process.get(), workingSetsSpan.data(), workingSetsSpan.size_bytes()))
            return countSizeOfPrivateWorkingSet(workingSetsSpan[0]);

        if (GetLastError() != ERROR_BAD_LENGTH)
            return 0;
        numberOfEntries = updateNumberOfEntries(workingSetsSpan[0].NumberOfEntries);
    }
}

}
