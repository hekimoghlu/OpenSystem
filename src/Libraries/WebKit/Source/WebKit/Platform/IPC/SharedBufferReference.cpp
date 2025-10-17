/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 28, 2024.
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
#include "SharedBufferReference.h"

#include "Decoder.h"
#include "Encoder.h"
#include <WebCore/SharedMemory.h>

namespace IPC {

using namespace WebCore;

#if !USE(UNIX_DOMAIN_SOCKETS)
SharedBufferReference::SharedBufferReference(std::optional<SerializableBuffer>&& serializableBuffer)
{
    if (!serializableBuffer)
        return;

    if (!serializableBuffer->size) {
        m_buffer = SharedBuffer::create();
        return;
    }

    if (!serializableBuffer->handle)
        return;

    auto sharedMemoryBuffer = SharedMemory::map(WTFMove(*serializableBuffer->handle), SharedMemory::Protection::ReadOnly);
    if (!sharedMemoryBuffer || sharedMemoryBuffer->size() < serializableBuffer->size)
        return;

    m_size = serializableBuffer->size;
    m_memory = WTFMove(sharedMemoryBuffer);
}

auto SharedBufferReference::serializableBuffer() const -> std::optional<SerializableBuffer>
{
    if (isNull())
        return std::nullopt;
    if (!m_size)
        return SerializableBuffer { 0, std::nullopt };
    auto sharedMemoryBuffer = m_memory ? m_memory : SharedMemory::copyBuffer(*m_buffer.copyRef());
    return SerializableBuffer { m_size, sharedMemoryBuffer->createHandle(SharedMemory::Protection::ReadOnly) };
}
#endif

RefPtr<WebCore::SharedBuffer> SharedBufferReference::unsafeBuffer() const
{
#if !USE(UNIX_DOMAIN_SOCKETS)
    RELEASE_ASSERT_WITH_MESSAGE(isEmpty() || (!m_buffer && m_memory), "Must only be called on IPC's receiver side");

    if (RefPtr memory = m_memory)
        return memory->createSharedBuffer(m_size);
#endif
    if (RefPtr buffer = m_buffer)
        return buffer->makeContiguous();
    return nullptr;
}

std::span<const uint8_t> SharedBufferReference::span() const
{
#if !USE(UNIX_DOMAIN_SOCKETS)
    RELEASE_ASSERT_WITH_MESSAGE(isEmpty() || (!m_buffer && m_memory), "Must only be called on IPC's receiver side");

    if (m_memory)
        return m_memory->span().first(m_size);
#endif
    if (!m_buffer)
        return { };

    if (!m_buffer->isContiguous())
        m_buffer = m_buffer->makeContiguous();

    return downcast<SharedBuffer>(m_buffer.get())->span().first(m_size);
}

RefPtr<WebCore::SharedMemory> SharedBufferReference::sharedCopy() const
{
    if (!m_size)
        return nullptr;
    return SharedMemory::copyBuffer(*unsafeBuffer());
}

} // namespace IPC
