/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 30, 2025.
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

// MallocPtr is a smart pointer class that calls fastFree in its destructor.
// It is intended to be used for pointers where the C++ lifetime semantics
// (calling constructors and destructors) is not desired. 

namespace WTF {

template<typename T, typename Malloc = FastMalloc> class MallocPtr {
    WTF_MAKE_NONCOPYABLE(MallocPtr);
public:
    MallocPtr() = default;

    constexpr MallocPtr(std::nullptr_t)
    {
    }

    MallocPtr(MallocPtr&& other)
        : m_ptr(other.leakPtr())
    {
    }

    ~MallocPtr()
    {
        Malloc::free(m_ptr);
    }

    T* get() const
    {
        return m_ptr;
    }

    T *leakPtr() WARN_UNUSED_RETURN
    {
        return std::exchange(m_ptr, nullptr);
    }

    explicit operator bool() const
    {
        return m_ptr;
    }

    bool operator!() const
    {
        return !m_ptr;
    }

    T& operator*() const
    {
        ASSERT(m_ptr);
        return *m_ptr;
    }

    T* operator->() const
    {
        return m_ptr;
    }

    MallocPtr& operator=(MallocPtr&& other)
    {
        MallocPtr ptr = WTFMove(other);
        swap(ptr);

        return *this;
    }

    void swap(MallocPtr& other)
    {
        std::swap(m_ptr, other.m_ptr);
    }

    template<typename U, typename OtherMalloc> friend MallocPtr<U, OtherMalloc> adoptMallocPtr(U*);

    static MallocPtr malloc(size_t size)
    {
        return MallocPtr {
            static_cast<T*>(Malloc::malloc(size))
        };
    }

    static MallocPtr zeroedMalloc(size_t size)
    {
        return MallocPtr {
            static_cast<T*>(Malloc::zeroedMalloc(size))
        };
    }

    static MallocPtr tryMalloc(size_t size)
    {
        return MallocPtr {
            static_cast<T*>(Malloc::tryMalloc(size))
        };
    }

    static MallocPtr tryZeroedMalloc(size_t size)
    {
        return MallocPtr {
            static_cast<T*>(Malloc::tryZeroedMalloc(size))
        };
    }

    void realloc(size_t newSize)
    {
        m_ptr = static_cast<T*>(Malloc::realloc(m_ptr, newSize));
    }

private:
    explicit MallocPtr(T* ptr)
        : m_ptr(ptr)
    {
    }

    T* m_ptr { nullptr };
};

static_assert(sizeof(MallocPtr<int>) == sizeof(int*));

template<typename U, typename OtherMalloc> MallocPtr<U, OtherMalloc> adoptMallocPtr(U* ptr)
{
    return MallocPtr<U, OtherMalloc>(ptr);
}

} // namespace WTF

using WTF::MallocPtr;
using WTF::adoptMallocPtr;
