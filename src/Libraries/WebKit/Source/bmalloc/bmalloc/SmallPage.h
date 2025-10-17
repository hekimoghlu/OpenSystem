/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 28, 2023.
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

#include "BAssert.h"
#include "List.h"
#include "Mutex.h"
#include "VMAllocate.h"
#include <mutex>

#if !BUSE(LIBPAS)

namespace bmalloc {

class SmallLine;

class SmallPage : public ListNode<SmallPage> {
public:
    void ref(UniqueLockHolder&);
    bool deref(UniqueLockHolder&);
    unsigned refCount(UniqueLockHolder&) { return m_refCount; }
    
    size_t sizeClass() { return m_sizeClass; }
    void setSizeClass(size_t sizeClass) { m_sizeClass = sizeClass; }
    
    bool hasFreeLines(UniqueLockHolder&) const { return m_hasFreeLines; }
    void setHasFreeLines(UniqueLockHolder&, bool hasFreeLines) { m_hasFreeLines = hasFreeLines; }
    
    bool hasPhysicalPages() { return m_hasPhysicalPages; }
    void setHasPhysicalPages(bool hasPhysicalPages) { m_hasPhysicalPages = hasPhysicalPages; }

    bool usedSinceLastScavenge() { return m_usedSinceLastScavenge; }
    void clearUsedSinceLastScavenge() { m_usedSinceLastScavenge = false; }
    void setUsedSinceLastScavenge() { m_usedSinceLastScavenge = true; }

    SmallLine* begin();

    unsigned char slide() const { return m_slide; }
    void setSlide(unsigned char slide) { m_slide = slide; }
    
private:
    unsigned char m_hasFreeLines: 1;
    unsigned char m_hasPhysicalPages: 1;
    unsigned char m_usedSinceLastScavenge: 1;
    unsigned char m_refCount: 7;
    unsigned char m_sizeClass;
    unsigned char m_slide;

static_assert(
    sizeClassCount <= std::numeric_limits<decltype(m_sizeClass)>::max(),
    "Largest size class must fit in SmallPage metadata");
};

using LineCache = std::array<List<SmallPage>, sizeClassCount>;

inline void SmallPage::ref(UniqueLockHolder&)
{
    BASSERT(!m_slide);
    ++m_refCount;
    BASSERT(m_refCount);
}

inline bool SmallPage::deref(UniqueLockHolder&)
{
    BASSERT(!m_slide);
    BASSERT(m_refCount);
    --m_refCount;
    return !m_refCount;
}

} // namespace bmalloc

#endif
