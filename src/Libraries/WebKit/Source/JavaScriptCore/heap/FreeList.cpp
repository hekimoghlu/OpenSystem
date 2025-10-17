/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 10, 2024.
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
#include "FreeList.h"

namespace JSC {

FreeList::FreeList(unsigned cellSize)
    : m_cellSize(cellSize)
{
}

FreeList::~FreeList()
{
}

void FreeList::clear()
{
    m_intervalStart = nullptr;
    m_intervalEnd = nullptr;
    m_nextInterval = std::bit_cast<FreeCell*>(static_cast<uintptr_t>(1));
    m_secret = 0;
    m_originalSize = 0;
}

void FreeList::initialize(FreeCell* start, uint64_t secret, unsigned bytes)
{
    if (UNLIKELY(!start)) {
        clear();
        return;
    }
    m_secret = secret;
    m_nextInterval = start;
    FreeCell::advance(m_secret, m_nextInterval, m_intervalStart, m_intervalEnd);
    m_originalSize = bytes;
}

void FreeList::dump(PrintStream& out) const
{
    out.print("{nextInterval = ", RawPointer(nextInterval()), ", secret = ", m_secret, ", intervalStart = ", RawPointer(m_intervalStart), ", intervalEnd = ", RawPointer(m_intervalEnd), ", originalSize = ", m_originalSize, "}");
}

} // namespace JSC

