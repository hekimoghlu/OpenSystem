/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 18, 2025.
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
// Encodes a FragmentedSharedBuffer that is to be sent over IPC
// WARNING: a SharedBufferReference should only be accessed on the IPC's receiver side.

#pragma once

#include <WebCore/SharedBuffer.h>
#include <WebCore/SharedMemory.h>
#include <optional>

namespace IPC {

class SharedBufferReference {
public:
    SharedBufferReference() = default;

    explicit SharedBufferReference(RefPtr<WebCore::FragmentedSharedBuffer>&& buffer)
        : m_size(buffer ? buffer->size() : 0)
        , m_buffer(WTFMove(buffer)) { }
    explicit SharedBufferReference(Ref<WebCore::FragmentedSharedBuffer>&& buffer)
        : m_size(buffer->size())
        , m_buffer(WTFMove(buffer)) { }
    explicit SharedBufferReference(RefPtr<WebCore::SharedBuffer>&& buffer)
        : m_size(buffer ? buffer->size() : 0)
        , m_buffer(WTFMove(buffer)) { }
    explicit SharedBufferReference(Ref<WebCore::SharedBuffer>&& buffer)
        : m_size(buffer->size())
        , m_buffer(WTFMove(buffer)) { }
    explicit SharedBufferReference(const WebCore::FragmentedSharedBuffer& buffer)
        : m_size(buffer.size())
        , m_buffer(const_cast<WebCore::FragmentedSharedBuffer*>(&buffer)) { }

#if !USE(UNIX_DOMAIN_SOCKETS)
    struct SerializableBuffer {
        size_t size;
        std::optional<WebCore::SharedMemory::Handle> handle;
    };
    SharedBufferReference(std::optional<SerializableBuffer>&&);
#endif

    SharedBufferReference(const SharedBufferReference&) = default;
    SharedBufferReference(SharedBufferReference&&) = default;
    SharedBufferReference& operator=(const SharedBufferReference&) = default;
    SharedBufferReference& operator=(SharedBufferReference&&) = default;

    size_t size() const { return m_size; }
    bool isEmpty() const { return !size(); }
    bool isNull() const { return isEmpty() && !m_buffer; }

#if USE(UNIX_DOMAIN_SOCKETS)
    RefPtr<WebCore::FragmentedSharedBuffer> buffer() const { return m_buffer; }
#else
    std::optional<SerializableBuffer> serializableBuffer() const;
#endif

    // The following method must only be used on the receiver's IPC side.
    // It relies on an implementation detail that makes m_buffer become a contiguous SharedBuffer
    // once it's deserialised over IPC.
    RefPtr<WebCore::SharedBuffer> unsafeBuffer() const;
    std::span<const uint8_t> span() const;
    RefPtr<WebCore::SharedMemory> sharedCopy() const;

private:
    SharedBufferReference(Ref<WebCore::SharedMemory>&& memory, size_t size)
        : m_size(size)
        , m_memory(WTFMove(memory)) { }

    size_t m_size { 0 };
    mutable RefPtr<WebCore::FragmentedSharedBuffer> m_buffer;
    RefPtr<WebCore::SharedMemory> m_memory; // Only set on the receiver side and if m_size isn't 0.
};

} // namespace IPC
