/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 11, 2024.
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

#include <wtf/ArgumentCoder.h>
#include <wtf/Hasher.h>
#include <wtf/ThreadSafeRefCounted.h>
#include <wtf/Vector.h>

namespace WebCore {

class ThreadSafeDataBuffer;

class ThreadSafeDataBufferImpl : public ThreadSafeRefCounted<ThreadSafeDataBufferImpl> {
private:
    friend class ThreadSafeDataBuffer;
    friend struct IPC::ArgumentCoder<ThreadSafeDataBufferImpl, void>;

    static Ref<ThreadSafeDataBufferImpl> create(Vector<uint8_t>&& data)
    {
        return adoptRef(*new ThreadSafeDataBufferImpl(WTFMove(data)));
    }

    ThreadSafeDataBufferImpl(Vector<uint8_t>&& data)
        : m_data(WTFMove(data))
    {
    }

    ThreadSafeDataBufferImpl(std::span<const uint8_t> data)
        : m_data(data)
    {
    }

    Vector<uint8_t> m_data;
};

class ThreadSafeDataBuffer {
private:
    friend struct IPC::ArgumentCoder<ThreadSafeDataBuffer, void>;
public:
    static ThreadSafeDataBuffer create(Vector<uint8_t>&& data)
    {
        return ThreadSafeDataBuffer(WTFMove(data));
    }

    static ThreadSafeDataBuffer copyData(std::span<const uint8_t> data)
    {
        return ThreadSafeDataBuffer(data);
    }

    ThreadSafeDataBuffer() = default;

    ThreadSafeDataBuffer isolatedCopy() const { return *this; }
    
    const Vector<uint8_t>* data() const
    {
        return m_impl ? &m_impl->m_data : nullptr;
    }

    size_t size() const
    {
        return m_impl ? m_impl->m_data.size() : 0;
    }

    bool operator==(const ThreadSafeDataBuffer& other) const
    {
        if (!m_impl)
            return !other.m_impl;

        return m_impl->m_data == other.m_impl->m_data;
    }

private:
    static ThreadSafeDataBuffer create(RefPtr<ThreadSafeDataBufferImpl>&& impl)
    {
        return ThreadSafeDataBuffer(WTFMove(impl));
    }

    explicit ThreadSafeDataBuffer(RefPtr<ThreadSafeDataBufferImpl>&& impl)
        : m_impl(WTFMove(impl))
    {
    }

    explicit ThreadSafeDataBuffer(Vector<uint8_t>&& data)
        : m_impl(adoptRef(new ThreadSafeDataBufferImpl(WTFMove(data))))
    {
    }

    explicit ThreadSafeDataBuffer(std::span<const uint8_t> data)
        : m_impl(adoptRef(new ThreadSafeDataBufferImpl(data)))
    {
    }

    RefPtr<ThreadSafeDataBufferImpl> m_impl;
};

inline void add(Hasher& hasher, const ThreadSafeDataBuffer& buffer)
{
    auto* data = buffer.data();
    if (!data) {
        add(hasher, true);
        return;
    }
    add(hasher, false);
    add(hasher, *data);
}

} // namespace WebCore
