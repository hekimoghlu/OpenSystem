/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 10, 2024.
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

#include <memory>
#include <wtf/Assertions.h>
#include <wtf/GetPtr.h>
#include <wtf/TypeCasts.h>

namespace WTF {

template<typename T> class UniqueRef;

template<typename T, class... Args>
UniqueRef<T> makeUniqueRefWithoutFastMallocCheck(Args&&... args)
{
    return UniqueRef<T>(*new T(std::forward<Args>(args)...));
}

template<class T, class... Args>
UniqueRef<T> makeUniqueRefWithoutRefCountedCheck(Args&&... args)
{
    static_assert(std::is_same<typename T::WTFIsFastAllocated, int>::value, "T should use FastMalloc (WTF_MAKE_FAST_ALLOCATED)");
    return makeUniqueRefWithoutFastMallocCheck<T>(std::forward<Args>(args)...);
}

template<typename T, class... Args>
UniqueRef<T> makeUniqueRef(Args&&... args)
{
    static_assert(std::is_same<typename T::WTFIsFastAllocated, int>::value, "T should use FastMalloc (WTF_MAKE_FAST_ALLOCATED)");
    static_assert(!HasRefPtrMemberFunctions<T>::value, "T should not be RefCounted");
    return makeUniqueRefWithoutFastMallocCheck<T>(std::forward<Args>(args)...);
}

template<typename T>
UniqueRef<T> makeUniqueRefFromNonNullUniquePtr(std::unique_ptr<T>&& ptr)
{
    return UniqueRef<T>(WTFMove(ptr));
}

template<typename T>
class UniqueRef {
public:
    template <typename U>
    UniqueRef(UniqueRef<U>&& other)
        : m_ref(other.m_ref.release())
    {
        ASSERT(m_ref);
    }

    explicit UniqueRef(T& other)
        : m_ref(&other)
    {
        ASSERT(m_ref);
    }

    T* ptr() const RETURNS_NONNULL { ASSERT(m_ref); return m_ref.get(); }
    T& get() const { ASSERT(m_ref); return *m_ref; }

    T* operator&() const { ASSERT(m_ref); return m_ref.get(); }
    T* operator->() const { ASSERT(m_ref); return m_ref.get(); }

    operator T&() const { ASSERT(m_ref); return *m_ref; }
    T& operator*() const { ASSERT(m_ref); return *m_ref.get(); }

    std::unique_ptr<T> moveToUniquePtr() { return WTFMove(m_ref); }

    explicit UniqueRef(HashTableEmptyValueType) { }
    bool isHashTableEmptyValue() const { return !m_ref; }

private:
    template<class U, class... Args> friend UniqueRef<U> makeUniqueRefWithoutFastMallocCheck(Args&&...);
    template<class U> friend UniqueRef<U> makeUniqueRefFromNonNullUniquePtr(std::unique_ptr<U>&&);
    template<class U> friend class UniqueRef;

    explicit UniqueRef(std::unique_ptr<T>&& ptr)
        : m_ref(WTFMove(ptr))
    {
        ASSERT(m_ref);
    }

    std::unique_ptr<T> m_ref;
};

template <typename T>
struct GetPtrHelper<UniqueRef<T>> {
    using PtrType = T*;
    using UnderlyingType = T;
    static T* getPtr(const UniqueRef<T>& p) { return const_cast<T*>(p.ptr()); }
};

template <typename T>
struct IsSmartPtr<UniqueRef<T>> {
    static constexpr bool value = true;
    static constexpr bool isNullable = false;
};

template<typename ExpectedType, typename ArgType>
inline bool is(UniqueRef<ArgType>& source)
{
    return is<ExpectedType>(source.get());
}

template<typename ExpectedType, typename ArgType>
inline bool is(const UniqueRef<ArgType>& source)
{
    return is<ExpectedType>(source.get());
}

} // namespace WTF

using WTF::UniqueRef;
using WTF::makeUniqueRef;
using WTF::makeUniqueRefWithoutFastMallocCheck;
using WTF::makeUniqueRefWithoutRefCountedCheck;
using WTF::makeUniqueRefFromNonNullUniquePtr;
