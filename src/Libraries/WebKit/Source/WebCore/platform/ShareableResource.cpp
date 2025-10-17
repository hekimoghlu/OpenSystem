/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 23, 2022.
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
#include "ShareableResource.h"

#if ENABLE(SHAREABLE_RESOURCE)

#include "SharedBuffer.h"
#include <wtf/CheckedArithmetic.h>

namespace WebCore {

ShareableResourceHandle::ShareableResourceHandle(SharedMemory::Handle&& handle, unsigned offset, unsigned size)
    : m_handle(WTFMove(handle))
    , m_offset(offset)
    , m_size(size)
{
}

RefPtr<SharedBuffer> ShareableResource::wrapInSharedBuffer()
{
    return SharedBuffer::create(DataSegment::Provider {
        [self = Ref { *this }]() { return self->span(); }
    });
}

RefPtr<SharedBuffer> ShareableResourceHandle::tryWrapInSharedBuffer() &&
{
    RefPtr<ShareableResource> resource = ShareableResource::map(WTFMove(*this));
    if (!resource) {
        LOG_ERROR("Failed to recreate ShareableResource from handle.");
        return nullptr;
    }

    return resource->wrapInSharedBuffer();
}

RefPtr<ShareableResource> ShareableResource::create(Ref<SharedMemory>&& sharedMemory, unsigned offset, unsigned size)
{
    auto totalSize = CheckedSize(offset) + size;
    if (totalSize.hasOverflowed()) {
        LOG_ERROR("Failed to create ShareableResource from SharedMemory due to overflow.");
        return nullptr;
    }
    if (totalSize > sharedMemory->size()) {
        LOG_ERROR("Failed to create ShareableResource from SharedMemory due to mismatched buffer size.");
        return nullptr;
    }
    return adoptRef(*new ShareableResource(WTFMove(sharedMemory), offset, size));
}

RefPtr<ShareableResource> ShareableResource::map(Handle&& handle)
{
    auto sharedMemory = SharedMemory::map(WTFMove(handle.m_handle), SharedMemory::Protection::ReadOnly);
    if (!sharedMemory)
        return nullptr;

    return create(sharedMemory.releaseNonNull(), handle.m_offset, handle.m_size);
}

ShareableResource::ShareableResource(Ref<SharedMemory>&& sharedMemory, unsigned offset, unsigned size)
    : m_sharedMemory(WTFMove(sharedMemory))
    , m_offset(offset)
    , m_size(size)
{
}

ShareableResource::~ShareableResource() = default;

auto ShareableResource::createHandle() -> std::optional<Handle>
{
    auto memoryHandle = m_sharedMemory->createHandle(SharedMemory::Protection::ReadOnly);
    if (!memoryHandle)
        return std::nullopt;

    return { Handle { WTFMove(*memoryHandle), m_offset, m_size } };
}

std::span<const uint8_t> ShareableResource::span() const
{
    return m_sharedMemory->span().subspan(m_offset, m_size);
}

unsigned ShareableResource::size() const
{
    return m_size;
}

} // namespace WebCore

#endif // ENABLE(SHAREABLE_RESOURCE)
