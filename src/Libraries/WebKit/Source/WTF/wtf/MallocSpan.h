/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 9, 2022.
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

#include <utility>
#include <wtf/FastMalloc.h>
#include <wtf/Noncopyable.h>
#include <wtf/StdLibExtras.h>
#include <wtf/TypeTraits.h>

// MallocSpan is a smart pointer class that wraps a std::span and calls fastFree in its destructor.

namespace WTF {

template<typename T, typename Malloc = FastMalloc> class MallocSpan {
    WTF_MAKE_NONCOPYABLE(MallocSpan);
public:
    MallocSpan() = default;

    MallocSpan(MallocSpan&& other)
        : m_span(other.leakSpan())
    {
    }

    template<typename U>
    MallocSpan(MallocSpan<U, Malloc>&& other) requires (std::is_same_v<T, uint8_t>)
        : m_span(asWritableBytes(other.leakSpan()))
    {
    }

    ~MallocSpan()
    {
        if constexpr (parameterCount(Malloc::free) == 2)
            Malloc::free(m_span.data(), m_span.size());
        else
            Malloc::free(m_span.data());
    }

    MallocSpan& operator=(MallocSpan&& other)
    {
        MallocSpan ptr = WTFMove(other);
        swap(ptr);
        return *this;
    }

    void swap(MallocSpan& other)
    {
        std::swap(m_span, other.m_span);
    }

    size_t sizeInBytes() const { return m_span.size_bytes(); }

    std::span<const T> span() const LIFETIME_BOUND { return spanConstCast<const T>(m_span); }
    std::span<T> mutableSpan() LIFETIME_BOUND { return m_span; }
    std::span<T> leakSpan() WARN_UNUSED_RETURN { return std::exchange(m_span, std::span<T>()); }

    T& operator[](size_t i) LIFETIME_BOUND { return m_span[i]; }
    const T& operator[](size_t i) const LIFETIME_BOUND { return m_span[i]; }

    explicit operator bool() const
    {
        return !!m_span.data();
    }

    bool operator!() const
    {
        return !m_span.data();
    }

    static MallocSpan malloc(size_t sizeInBytes)
    {
        return MallocSpan { static_cast<T*>(Malloc::malloc(sizeInBytes)), sizeInBytes };
    }

    static MallocSpan zeroedMalloc(size_t sizeInBytes)
    {
        return MallocSpan { static_cast<T*>(Malloc::zeroedMalloc(sizeInBytes)), sizeInBytes };
    }

    static MallocSpan alignedMalloc(size_t alignment, size_t sizeInBytes)
    {
        return MallocSpan { static_cast<T*>(Malloc::alignedMalloc(alignment, sizeInBytes)), sizeInBytes };
    }

#if HAVE(MMAP)
    static MallocSpan mmap(size_t sizeInBytes, int pageProtection, int options, int fileDescriptor)
    {
        return MallocSpan { static_cast<T*>(Malloc::mmap(sizeInBytes, pageProtection, options, fileDescriptor)), sizeInBytes };
    }
#endif

    static MallocSpan tryMalloc(size_t sizeInBytes)
    {
        auto* ptr = Malloc::tryMalloc(sizeInBytes);
        if (!ptr)
            return { };
        return MallocSpan { static_cast<T*>(ptr), sizeInBytes };
    }

    static MallocSpan tryZeroedMalloc(size_t sizeInBytes)
    {
        auto* ptr = Malloc::tryZeroedMalloc(sizeInBytes);
        if (!ptr)
            return { };
        return MallocSpan { static_cast<T*>(ptr), sizeInBytes };
    }

    static MallocSpan tryAlignedMalloc(size_t alignment, size_t sizeInBytes)
    {
        auto* ptr = Malloc::tryAlignedMalloc(alignment, sizeInBytes);
        if (!ptr)
            return { };
        return MallocSpan { static_cast<T*>(ptr), sizeInBytes };
    }

    void realloc(size_t newSizeInBytes)
    {
        RELEASE_ASSERT(!(newSizeInBytes % sizeof(T)));
        m_span = unsafeMakeSpan(static_cast<T*>(Malloc::realloc(m_span.data(), newSizeInBytes)), newSizeInBytes / sizeof(T));
    }

private:
    template<typename U, typename OtherMalloc> friend MallocSpan<U, OtherMalloc> adoptMallocSpan(std::span<U>);

    explicit MallocSpan(T* ptr, size_t sizeInBytes)
        : m_span(unsafeMakeSpan(ptr, sizeInBytes / sizeof(T)))
    {
        RELEASE_ASSERT(!(sizeInBytes % sizeof(T)));
    }

    std::span<T> m_span;
};

template<typename U, typename OtherMalloc> MallocSpan<U, OtherMalloc> adoptMallocSpan(std::span<U> span)
{
    return MallocSpan<U, OtherMalloc>(span.data(), span.size_bytes());
}

} // namespace WTF

using WTF::MallocSpan;
using WTF::adoptMallocSpan;
