/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 4, 2023.
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
#include "HeapConstants.h"
#include <algorithm>

#if !BUSE(LIBPAS)

namespace bmalloc {

DEFINE_STATIC_PER_PROCESS_STORAGE(HeapConstants);

HeapConstants::HeapConstants(const LockHolder&)
    : m_vmPageSizePhysical { vmPageSizePhysical() }
{
    RELEASE_BASSERT(m_vmPageSizePhysical >= smallPageSize);
    RELEASE_BASSERT(vmPageSize() >= m_vmPageSizePhysical);

    initializeLineMetadata();
    initializePageMetadata();
}

template <class C>
constexpr void fillLineMetadata(C& container, size_t VMPageSize)
{
    constexpr size_t clsCount = sizeClass(smallLineSize);
    size_t lineCount = smallLineCount(VMPageSize);

    for (size_t cls = 0; cls < clsCount; ++cls) {
        size_t size = objectSize(cls);
        size_t baseIndex = cls * lineCount;
        size_t object = 0;
        while (object < VMPageSize) {
            size_t line = object / smallLineSize;
            size_t leftover = object % smallLineSize;

            auto objectCount = divideRoundingUp(smallLineSize - leftover, size);

            object += objectCount * size;

            // Don't allow the last object in a page to escape the page.
            if (object > VMPageSize) {
                BASSERT(objectCount);
                --objectCount;
            }

            container[baseIndex + line] = { static_cast<unsigned char>(leftover), static_cast<unsigned char>(objectCount) };
        }
    }
}

template <size_t VMPageSize>
constexpr auto computeLineMetadata()
{
    std::array<LineMetadata, sizeClass(smallLineSize) * smallLineCount(VMPageSize)> result;
    fillLineMetadata(result, VMPageSize);
    return result;
}

#if BUSE(PRECOMPUTED_CONSTANTS_VMPAGE4K)
constexpr auto kPrecalcuratedLineMetadata4k = computeLineMetadata<4 * kB>();
#endif

#if BUSE(PRECOMPUTED_CONSTANTS_VMPAGE16K)
constexpr auto kPrecalcuratedLineMetadata16k = computeLineMetadata<16 * kB>();
#endif

void HeapConstants::initializeLineMetadata()
{
#if BUSE(PRECOMPUTED_CONSTANTS_VMPAGE4K)
    if (m_vmPageSizePhysical == 4 * kB) {
        m_smallLineMetadata = &kPrecalcuratedLineMetadata4k[0];
        return;
    }
#endif

#if BUSE(PRECOMPUTED_CONSTANTS_VMPAGE16K)
    if (m_vmPageSizePhysical == 16 * kB) {
        m_smallLineMetadata = &kPrecalcuratedLineMetadata16k[0];
        return;
    }
#endif

    size_t sizeClassCount = bmalloc::sizeClass(smallLineSize);
    m_smallLineMetadataStorage.grow(sizeClassCount * smallLineCount());
    fillLineMetadata(m_smallLineMetadataStorage, m_vmPageSizePhysical);
    m_smallLineMetadata = &m_smallLineMetadataStorage[0];
}

void HeapConstants::initializePageMetadata()
{
    auto computePageSize = [&](size_t sizeClass) {
        size_t size = objectSize(sizeClass);
        if (sizeClass < bmalloc::sizeClass(smallLineSize))
            return m_vmPageSizePhysical;

        for (size_t pageSize = m_vmPageSizePhysical; pageSize < pageSizeMax; pageSize += m_vmPageSizePhysical) {
            RELEASE_BASSERT(pageSize <= chunkSize / 2);
            size_t waste = pageSize % size;
            if (waste <= pageSize / pageSizeWasteFactor)
                return pageSize;
        }

        return pageSizeMax;
    };

    for (size_t i = 0; i < sizeClassCount; ++i)
        m_pageClasses[i] = (computePageSize(i) - 1) / smallPageSize;
}

} // namespace bmalloc

#endif
