/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 2, 2023.
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

#include "ArgumentCoders.h"
#include "MessageNames.h"
#include <wtf/StdLibExtras.h>

namespace IPC {

template<typename, typename> struct ArgumentCoder;

// IPC encoder which:
//  - Encodes to a caller buffer with fixed size, does not resize, stops when size runs out
//  - Does not initialize alignment gaps
//
class StreamConnectionEncoder final {
public:
    // Stream allocation needs to be at least size of StreamSetDestinationID message at any offset % messageAlignment.
    // StreamSetDestinationID has MessageName+uint64_t, where uint64_t is expected to to be aligned at 8.
    static constexpr size_t minimumMessageSize = 16;
    static constexpr size_t messageAlignment = alignof(MessageName);
    static constexpr bool isIPCEncoder = true;

    StreamConnectionEncoder(MessageName messageName, std::span<uint8_t> stream)
        : m_buffer(stream)
    {
        *this << messageName;
    }

    ~StreamConnectionEncoder() = default;

    template<typename T, size_t Extent>
    bool encodeSpan(std::span<T, Extent> span)
    {
        auto bytes = asBytes(span);
        auto bufferPointer = reinterpret_cast<uintptr_t>(m_buffer.data()) + m_encodedSize;
        auto newBufferPointer = roundUpToMultipleOf<alignof(T)>(bufferPointer);
        if (newBufferPointer < bufferPointer)
            return false;
        auto alignedSize = m_encodedSize + (newBufferPointer - bufferPointer);
        if (!reserve(alignedSize, bytes.size()))
            return false;
        memcpySpan(m_buffer.subspan(alignedSize), bytes);
        m_encodedSize = alignedSize + bytes.size();
        return true;
    }

    template<typename T>
    bool encodeObject(const T& object)
    {
        static_assert(std::is_trivially_copyable_v<T>);
        return encodeSpan(singleElementSpan(object));
    }

    template<typename T>
    StreamConnectionEncoder& operator<<(T&& t)
    {
        ArgumentCoder<std::remove_cvref_t<T>, void>::encode(*this, std::forward<T>(t));
        return *this;
    }

    size_t size() const { ASSERT(isValid()); return m_encodedSize; }
    bool isValid() const { return !!m_buffer.data(); }
    operator bool() const { return isValid(); }
private:
    bool reserve(size_t alignedSize, size_t additionalSize)
    {
        size_t size = alignedSize + additionalSize;
        if (size < alignedSize || size > m_buffer.size()) {
            m_buffer = { };
            return false;
        }
        return true;
    }
    std::span<uint8_t> m_buffer;
    size_t m_encodedSize { 0 };
};

} // namespace IPC
