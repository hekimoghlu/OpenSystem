/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 17, 2024.
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

#include <wtf/OSAllocator.h>

namespace JSC {

// This class implements a simple array class that can be grown by appending items to the end.
// This class is implemented purely in terms of system allocations, with no malloc/free, so that
// it can safely be used from a secondary thread whilst the main thrad is paused (potentially
// holding the fast malloc heap lock).
template<typename T>
class TieredMMapArray {
    static const size_t entriesPerBlock = 4096;

public:
    TieredMMapArray()
        : m_directoryCount(4096)
        , m_directory(static_cast<T**>(OSAllocator::reserveAndCommit(m_directoryCount * sizeof(T*))))
        , m_size(0)
    {
        for (size_t block = 0; block < m_directoryCount; ++block)
            m_directory[block] = 0;
    }

    ~TieredMMapArray()
    {
        size_t usedCount = (m_size + (entriesPerBlock - 1)) / entriesPerBlock;
        ASSERT(usedCount == m_directoryCount || !m_directory[usedCount]);

        for (size_t block = 0; block < usedCount; ++block) {
            ASSERT(m_directory[block]);
            OSAllocator::decommitAndRelease(m_directory[block], entriesPerBlock * sizeof(T));
        }

        OSAllocator::decommitAndRelease(m_directory, m_directoryCount * sizeof(T*));
    }

    T& operator[](size_t index)
    {
        ASSERT(index < m_size);
        size_t block = index / entriesPerBlock;
        size_t offset = index % entriesPerBlock;

        ASSERT(m_directory[block]);
        return m_directory[block][offset];
    }

    void append(const T& value)
    {
        // Check if the array is completely full, if so create more capacity in the directory.
        if (m_size == m_directoryCount * entriesPerBlock) {
            // Reallocate the directory.
            size_t oldDirectorySize = m_directoryCount * sizeof(T*);
            size_t newDirectorySize = oldDirectorySize * 2;
            RELEASE_ASSERT(newDirectorySize < oldDirectorySize);
            m_directory = OSAllocator::reallocateCommitted(m_directory, oldDirectorySize, newDirectorySize);

            // 
            size_t newDirectoryCount = m_directoryCount * 2;
            for (size_t block = m_directoryCount; block < newDirectoryCount; ++block)
                m_directory[block] = 0;
            m_directoryCount = newDirectoryCount;
        }

        size_t index = m_size;
        size_t block = index / entriesPerBlock;
        size_t offset = index % entriesPerBlock;

        if (!offset) {
            ASSERT(!m_directory[block]);
            m_directory[block] = static_cast<T*>(OSAllocator::reserveAndCommit(entriesPerBlock * sizeof(T)));
        }

        ASSERT(m_directory[block]);
        ++m_size;
        m_directory[block][offset] = value;
    }

    size_t size() const { return m_size; }

private:
    size_t m_directoryCount;
    T** m_directory;
    size_t m_size;
};

} // namespace JSC
