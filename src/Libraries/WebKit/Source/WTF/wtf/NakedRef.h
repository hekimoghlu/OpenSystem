/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 31, 2022.
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

#include <wtf/FastMalloc.h>
#include <wtf/RawPtrTraits.h>

namespace WTF {

// The purpose of this class is to ensure that the wrapped pointer will never be
// used uninitialized.

template <typename T> class NakedRef {
    WTF_MAKE_FAST_ALLOCATED;
public:
    ALWAYS_INLINE NakedRef(T& ref) : m_ptr(&ref) { }
    ALWAYS_INLINE NakedRef(const NakedRef&) = delete;
    template<typename U> NakedRef(const NakedRef<U>&) = delete;

    ALWAYS_INLINE NakedRef(NakedRef&& other)
        : m_ptr(&other.leakRef())
    {
        ASSERT(m_ptr);
    }

    template<typename U>
    NakedRef(NakedRef<U>&& other)
        : m_ptr(&other.leakRef())
    {
        ASSERT(m_ptr);
    }

    ALWAYS_INLINE T* operator->() const { ASSERT(m_ptr); return m_ptr; }
    T* ptr() const RETURNS_NONNULL { ASSERT(m_ptr); return m_ptr; }
    T& get() const { return *m_ptr; }
    operator T&() const { ASSERT(m_ptr); return *m_ptr; }
    bool operator!() const { ASSERT(m_ptr); return !*m_ptr; }

    NakedRef copyRef() && = delete;
    NakedRef copyRef() const & WARN_UNUSED_RETURN { return NakedRef(*m_ptr); }

    NakedRef& operator=(T&);
    NakedRef& operator=(NakedRef&&);
    template<typename U> NakedRef& operator=(NakedRef<U>&&);

    // Use copyRef() and the move assignment operators instead.
    NakedRef& operator=(const NakedRef&) = delete;
    template<typename X> NakedRef& operator=(const NakedRef<X>&) = delete;

    template<typename U> void swap(NakedRef<U>&);

    T& leakRef() WARN_UNUSED_RETURN
    {
        ASSERT(m_ptr);
        T& result = *RawPtrTraits<T>::exchange(m_ptr, nullptr);
        return result;
    }

private:
    T* m_ptr;
};

template<typename T> inline NakedRef<T>& NakedRef<T>::operator=(NakedRef&& reference)
{
    NakedRef movedReference = WTFMove(reference);
    swap(movedReference);
    return *this;
}

template<typename T> inline NakedRef<T>& NakedRef<T>::operator=(T& ref)
{
    NakedRef copiedReference = ref;
    swap(copiedReference);
    return *this;
}

template<typename T> template<typename U> inline NakedRef<T>& NakedRef<T>::operator=(NakedRef<U>&& other)
{
    NakedRef ref = WTFMove(other);
    swap(ref);
    return *this;
}

template<class T>
template<class U>
inline void NakedRef<T>::swap(NakedRef<U>& other)
{
    std::swap(m_ptr, other.m_ptr);
}

template<class T, class U> inline void swap(NakedRef<T>& a, NakedRef<U>& b)
{
    a.swap(b);
}

} // namespace WTF

using WTF::NakedRef;
