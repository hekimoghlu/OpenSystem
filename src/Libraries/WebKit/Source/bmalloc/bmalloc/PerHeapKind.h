/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 2, 2022.
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

#include "HeapKind.h"
#include <array>

#if !BUSE(LIBPAS)

namespace bmalloc {

template<typename T>
class PerHeapKindBase {
public:
    PerHeapKindBase(const PerHeapKindBase&) = delete;
    PerHeapKindBase& operator=(const PerHeapKindBase&) = delete;
    
    template<typename... Arguments>
    PerHeapKindBase(Arguments&&... arguments)
    {
        for (unsigned i = numHeaps; i--;)
            new (&at(i)) T(static_cast<HeapKind>(i), std::forward<Arguments>(arguments)...);
    }
    
    static size_t size() { return numHeaps; }
    
    T& at(size_t i)
    {
        return *reinterpret_cast<T*>(&m_memory[i]);
    }
    
    const T& at(size_t i) const
    {
        return *reinterpret_cast<T*>(&m_memory[i]);
    }
    
    T& at(HeapKind heapKind)
    {
        return at(static_cast<size_t>(heapKind));
    }
    
    const T& at(HeapKind heapKind) const
    {
        return at(static_cast<size_t>(heapKind));
    }
    
    T& operator[](size_t i) { return at(i); }
    const T& operator[](size_t i) const { return at(i); }
    T& operator[](HeapKind heapKind) { return at(heapKind); }
    const T& operator[](HeapKind heapKind) const { return at(heapKind); }

private:
    BALLOW_DEPRECATED_DECLARATIONS_BEGIN
    typedef typename std::array<typename std::aligned_storage<sizeof(T), std::alignment_of<T>::value>::type, numHeaps> Memory;
    BALLOW_DEPRECATED_DECLARATIONS_END
    Memory m_memory;
};

template<typename T>
class StaticPerHeapKind : public PerHeapKindBase<T> {
public:
    template<typename... Arguments>
    StaticPerHeapKind(Arguments&&... arguments)
        : PerHeapKindBase<T>(std::forward<Arguments>(arguments)...)
    {
    }
    
    ~StaticPerHeapKind() = delete;
};

template<typename T>
class PerHeapKind : public PerHeapKindBase<T> {
public:
    template<typename... Arguments>
    PerHeapKind(Arguments&&... arguments)
        : PerHeapKindBase<T>(std::forward<Arguments>(arguments)...)
    {
    }
    
    ~PerHeapKind()
    {
        for (unsigned i = numHeaps; i--;)
            this->at(i).~T();
    }
};

} // namespace bmalloc

#endif
