/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 12, 2025.
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
#include "SharedBuffer.h"

#include <wtf/OSAllocator.h>
#include <wtf/cf/TypeCastsCF.h>

namespace WebCore {

FragmentedSharedBuffer::FragmentedSharedBuffer(CFDataRef data)
{
    append(data);
}

// Using Foundation allows for an even more efficient implementation of this function,
// so only use this version for non-Foundation.
#if !USE(FOUNDATION)
RetainPtr<CFDataRef> SharedBuffer::createCFData() const
{
    if (hasOneSegment()) {
        if (auto* data = std::get_if<RetainPtr<CFDataRef>>(&m_segments[0].segment->m_immutableData))
            return *data;
    }
    return adoptCF(CFDataCreate(nullptr, data(), size()));
}
#endif

Ref<FragmentedSharedBuffer> FragmentedSharedBuffer::create(CFDataRef data)
{
    return adoptRef(*new FragmentedSharedBuffer(data));
}

void FragmentedSharedBuffer::hintMemoryNotNeededSoon() const
{
    for (const auto& entry : m_segments) {
        if (entry.segment->hasOneRef()) {
            if (auto* data = std::get_if<RetainPtr<CFDataRef>>(&entry.segment->m_immutableData))
                OSAllocator::hintMemoryNotNeededSoon(const_cast<UInt8*>(CFDataGetBytePtr(data->get())), CFDataGetLength(data->get()));
        }
    }
}

void FragmentedSharedBuffer::append(CFDataRef data)
{
    ASSERT(!m_contiguous);
    if (data) {
        m_segments.append({m_size, DataSegment::create(data)});
        m_size += CFDataGetLength(data);
    }
    ASSERT(internallyConsistent());
}

}
