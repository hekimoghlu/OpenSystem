/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 7, 2022.
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
#include "SharedMemory.h"

#include "SharedBuffer.h"
#include <wtf/StdLibExtras.h>
#include <wtf/text/ParsingUtilities.h>

namespace WebCore {

bool isMemoryAttributionDisabled()
{
    static bool result = []() {
        auto value = unsafeSpan(getenv("WEBKIT_DISABLE_MEMORY_ATTRIBUTION"));
        if (!value.data())
            return false;
        return equalSpans(value, "1"_span);
    }();
    return result;
}

SharedMemoryHandle::SharedMemoryHandle(SharedMemoryHandle::Type&& handle, size_t size)
    : m_handle(WTFMove(handle))
    , m_size(size)
{
    RELEASE_ASSERT(!!m_handle);
}

RefPtr<SharedMemory> SharedMemory::copyBuffer(const FragmentedSharedBuffer& buffer)
{
    if (buffer.isEmpty())
        return nullptr;

    auto sharedMemory = allocate(buffer.size());
    if (!sharedMemory)
        return nullptr;

    auto destination = sharedMemory->mutableSpan();
    buffer.forEachSegment([&] (std::span<const uint8_t> segment) mutable {
        memcpySpan(consumeSpan(destination, segment.size()), segment);
    });

    return sharedMemory;
}

RefPtr<SharedMemory> SharedMemory::copySpan(std::span<const uint8_t> span)
{
    if (!span.size())
        return nullptr;

    auto sharedMemory = allocate(span.size());
    if (!sharedMemory)
        return nullptr;

    memcpySpan(sharedMemory->mutableSpan(), span);
    return sharedMemory;
}

Ref<SharedBuffer> SharedMemory::createSharedBuffer(size_t dataSize) const
{
    ASSERT(dataSize <= size());
    return SharedBuffer::create(DataSegment::Provider {
        [protectedThis = Ref { *this }, dataSize]() {
            return protectedThis->span().first(dataSize);
        }
    });
}

#if !PLATFORM(COCOA)
void SharedMemoryHandle::takeOwnershipOfMemory(MemoryLedger) const
{
}

void SharedMemoryHandle::setOwnershipOfMemory(const ProcessIdentity&, MemoryLedger) const
{
}
#endif

} // namespace WebCore
