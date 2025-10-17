/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 18, 2025.
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
#include "Mutex.h"
#include "ObjectType.h"
#include <mutex>

#if !BUSE(LIBPAS)

namespace bmalloc {

class SmallLine {
public:
    void ref(UniqueLockHolder&, unsigned char = 1);
    bool deref(UniqueLockHolder&);
    unsigned refCount(UniqueLockHolder&) { return m_refCount; }
    
    char* begin();
    char* end();

private:
    unsigned char m_refCount;

static_assert(
    smallLineSize / alignment <= std::numeric_limits<decltype(m_refCount)>::max(),
    "maximum object count must fit in SmallLine::m_refCount");

};

inline void SmallLine::ref(UniqueLockHolder&, unsigned char refCount)
{
    BASSERT(!m_refCount);
    m_refCount = refCount;
}

inline bool SmallLine::deref(UniqueLockHolder&)
{
    BASSERT(m_refCount);
    --m_refCount;
    return !m_refCount;
}

} // namespace bmalloc

#endif
