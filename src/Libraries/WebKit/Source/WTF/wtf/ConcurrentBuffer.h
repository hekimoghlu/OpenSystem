/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 23, 2022.
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

#include <wtf/Atomics.h>
#include <wtf/FastMalloc.h>
#include <wtf/HashFunctions.h>
#include <wtf/Lock.h>
#include <wtf/Noncopyable.h>
#include <wtf/StdLibExtras.h>
#include <wtf/Vector.h>

WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN

namespace WTF {

DECLARE_ALLOCATOR_WITH_HEAP_IDENTIFIER(ConcurrentBuffer);

// ConcurrentBuffer is suitable for when you plan to store immutable data and sometimes append to it.
// It supports storing data that is not copy-constructable but bit-copyable.
template<typename T>
class ConcurrentBuffer final {
    WTF_MAKE_NONCOPYABLE(ConcurrentBuffer);
    WTF_MAKE_FAST_ALLOCATED;
public:
    
    ConcurrentBuffer()
    {
    }
    
    ~ConcurrentBuffer()
    {
        if (Array* array = m_array) {
            for (size_t i = 0; i < array->size; ++i)
                array->data[i].~T();
        }
        for (Array* array : m_allArrays)
            ConcurrentBufferMalloc::free(array);
    }

    // Growing is not concurrent. This assumes you are holding some other lock before you do this.
    void growExact(size_t newSize)
    {
        Array* array = m_array;
        if (array && newSize <= array->size)
            return;
        Array* newArray = createArray(newSize);
        // This allows us to do ConcurrentBuffer<std::unique_ptr<>>.
        // asMutableByteSpan() avoids triggering -Wclass-memaccess.
        if (array)
            memcpySpan(asMutableByteSpan(newArray->span()), asByteSpan(array->span()));
        for (size_t i = array ? array->size : 0; i < newSize; ++i)
            new (newArray->data + i) T();
        WTF::storeStoreFence();
        m_array = newArray;
        WTF::storeStoreFence();
        m_allArrays.append(newArray);
    }
    
    void grow(size_t newSize)
    {
        size_t size = m_array ? m_array->size : 0;
        if (newSize <= size)
            return;
        growExact(std::max(newSize, size * 2));
    }
    
    struct Array {
        size_t size; // This is an immutable size.
        T data[1];

        std::span<T> span() { return unsafeMakeSpan(data, size); }
        std::span<const T> span() const { return unsafeMakeSpan(data, size); }
    };
    
    Array* array() const { return m_array; }
    
    T& operator[](size_t index) { return m_array->data[index]; }
    const T& operator[](size_t index) const { return m_array->data[index]; }
    
private:
    Array* createArray(size_t size)
    {
        Checked<size_t> objectSize = sizeof(T);
        objectSize *= size;
        objectSize += static_cast<size_t>(OBJECT_OFFSETOF(Array, data));
        Array* result = static_cast<Array*>(ConcurrentBufferMalloc::malloc(objectSize));
        result->size = size;
        return result;
    }
    
    Array* m_array { nullptr };
    Vector<Array*> m_allArrays;
};

} // namespace WTF

WTF_ALLOW_UNSAFE_BUFFER_USAGE_END
