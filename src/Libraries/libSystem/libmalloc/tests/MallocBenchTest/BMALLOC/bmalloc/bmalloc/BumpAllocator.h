/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 3, 2023.
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
#ifndef BumpAllocator_h
#define BumpAllocator_h

#include "BAssert.h"
#include "BumpRange.h"
#include "ObjectType.h"

namespace bmalloc {

// Helper object for allocating small objects.

class BumpAllocator {
public:
    BumpAllocator();
    void init(size_t);
    
    size_t size() { return m_size; }
    
    bool isNull() { return !m_ptr; }
    void clear();

    bool canAllocate() { return !!m_remaining; }
    void* allocate();

    void refill(const BumpRange&);

private:
    char* m_ptr;
    unsigned m_size;
    unsigned m_remaining;
};

inline BumpAllocator::BumpAllocator()
    : m_ptr()
    , m_size()
    , m_remaining()
{
}

inline void BumpAllocator::init(size_t size)
{
    m_ptr = nullptr;
    m_size = size;
    m_remaining = 0;
}

inline void* BumpAllocator::allocate()
{
    BASSERT(m_remaining);

    --m_remaining;
    char* result = m_ptr;
    m_ptr += m_size;
    return result;
}

inline void BumpAllocator::refill(const BumpRange& bumpRange)
{
    BASSERT(!canAllocate());
    m_ptr = bumpRange.begin;
    m_remaining = bumpRange.objectCount;
    BASSERT(canAllocate());
}

inline void BumpAllocator::clear()
{
    m_ptr = nullptr;
    m_remaining = 0;
}

} // namespace bmalloc

#endif // BumpAllocator_h
