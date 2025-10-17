/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 24, 2022.
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

#include <cstdint>
#include <utility>
#include <wtf/PtrTag.h>
#include <wtf/StdLibExtras.h>

namespace WTF {

template<typename T, const uintptr_t Tag>
class SignedPtr {
public:
    WTF_MAKE_FAST_ALLOCATED;
public:
    constexpr SignedPtr()
        : m_value(nullptr)
    {
    }

    constexpr SignedPtr(std::nullptr_t)
        : m_value(nullptr)
    {
    }

    SignedPtr(T* value)
    {
        set(value);
    }

    T* get() const
    {
#if CPU(ARM64E)
        if (!m_value)
            return nullptr;
        return ptrauth_auth_data(m_value, ptrauth_key_process_dependent_data, Tag);
#else
        return m_value;
#endif
    }

    void set(T* passedValue)
    {
#if CPU(ARM64E)
        if (!passedValue)
            return;
        m_value = ptrauth_sign_unauthenticated(passedValue, ptrauth_key_process_dependent_data, Tag);
#else
        m_value = passedValue;
#endif
    }

    void clear()
    {
        set(nullptr);
    }

    T* operator->() const { return get(); }

    template <typename U = T>
    typename std::enable_if<!std::is_void_v<U>, U&>::type operator*() const { return *get(); }

    bool operator!() const { return !m_value; }
    
    // This conversion operator allows implicit conversion to bool but not to other integer types.
    typedef T* (SignedPtr::*UnspecifiedBoolType);
    operator UnspecifiedBoolType() const { return get() ? &SignedPtr::m_value : nullptr; }
    explicit operator bool() const { return get(); }

    SignedPtr& operator=(T* value)
    {
        set(value);
        return *this;
    }

    template<class U>
    T* exchange(U&& newValue)
    {
        T* oldValue = get();
        set(std::forward<U>(newValue));
        return oldValue;
    }

    void swap(std::nullptr_t) { clear(); }

    void swap(SignedPtr& other)
    {
        T* t1 = get();
        T* t2 = other.get();
        set(t2);
        other.set(t1);
    }

private:
    T* m_value;
};

template <typename T, uintptr_t Tag>
struct IsSmartPtr<SignedPtr<T, Tag>> {
    static constexpr bool value = true;
    static constexpr bool isNullable = true;
};

template<typename T, uintptr_t Tag>
struct SignedPtrTraits {
    template<typename U, uintptr_t V> using RebindTraits = SignedPtrTraits<U, V>;

    using StorageType = SignedPtr<T, Tag>;

    template<class U> static ALWAYS_INLINE T* exchange(StorageType& ptr, U&& newValue) { return ptr.exchange(newValue); }
    template<typename Other> static ALWAYS_INLINE void swap(StorageType& a, Other& b) { a.swap(b); }

    static ALWAYS_INLINE T* unwrap(const StorageType& ptr) { return ptr.get(); }

    static StorageType hashTableDeletedValue() { return std::bit_cast<StorageType>(static_cast<uintptr_t>(-1)); }
    static ALWAYS_INLINE bool isHashTableDeletedValue(const StorageType& ptr) { return ptr == hashTableDeletedValue(); }
};

} // namespace WTF

using WTF::SignedPtrTraits;

