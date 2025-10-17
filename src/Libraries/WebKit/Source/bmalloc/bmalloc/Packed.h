/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 22, 2021.
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

#include "Algorithm.h"
#include <array>
#include <bit>

#if BOS(DARWIN)
#include <mach/vm_param.h>
#endif

#if !BUSE(LIBPAS)

namespace bmalloc {

template<typename T>
class Packed {
public:
    static_assert(std::is_trivial<T>::value);
    static constexpr bool isPackedType = true;

    Packed()
        : Packed(T { })
    {
    }

    Packed(const T& value)
    {
        memcpy(m_storage.data(), &value, sizeof(T));
    }

    T get() const
    {
        T value { };
        memcpy(&value, m_storage.data(), sizeof(T));
        return value;
    }

    void set(const T& value)
    {
        memcpy(m_storage.data(), &value, sizeof(T));
    }

    Packed<T>& operator=(const T& value)
    {
        set(value);
        return *this;
    }

    template<class U>
    T exchange(U&& newValue)
    {
        T oldValue = get();
        set(std::forward<U>(newValue));
        return oldValue;
    }

    void swap(Packed& other)
    {
        m_storage.swap(other.m_storage);
    }

    template<typename Other, typename = std::enable_if_t<Other::isPackedType>>
    void swap(Other& other)
    {
        T t1 = get();
        T t2 = other.get();
        set(t2);
        other.set(t1);
    }

    void swap(T& t2)
    {
        T t1 = get();
        std::swap(t1, t2);
        set(t1);
    }

private:
    std::array<uint8_t, sizeof(T)> m_storage;
};

// PackedAlignedPtr can take alignment parameter too. PackedAlignedPtr only uses this alignment information if it is profitable: we use
// alignment information only when we can reduce the size of the storage.
template<typename T, size_t alignment = alignof(T)>
class PackedAlignedPtr {
public:
    static_assert(isPowerOfTwo(alignment), "Alignment needs to be power-of-two");
    static constexpr bool isPackedType = true;
    static constexpr unsigned alignmentShiftSizeIfProfitable = getLSBSetNonZeroConstexpr(alignment);
    static constexpr unsigned storageSizeWithoutAlignmentShift = roundUpToMultipleOf<8, uintptr_t>(BOS_EFFECTIVE_ADDRESS_WIDTH) / 8;
    static constexpr unsigned storageSizeWithAlignmentShift = roundUpToMultipleOf<8, uintptr_t>(BOS_EFFECTIVE_ADDRESS_WIDTH - alignmentShiftSizeIfProfitable) / 8;
    static constexpr bool isAlignmentShiftProfitable = storageSizeWithoutAlignmentShift > storageSizeWithAlignmentShift;
    static constexpr unsigned alignmentShiftSize = isAlignmentShiftProfitable ? alignmentShiftSizeIfProfitable : 0;
    static constexpr unsigned storageSize = storageSizeWithAlignmentShift;

    constexpr PackedAlignedPtr()
        : m_storage()
    {
    }

    constexpr PackedAlignedPtr(std::nullptr_t)
        : m_storage()
    {
    }

    PackedAlignedPtr(T* value)
    {
        set(value);
    }

    T* get() const
    {
        // FIXME: PackedPtr<> can load memory with one mov by checking page boundary.
        // https://bugs.webkit.org/show_bug.cgi?id=197754
        uintptr_t value = 0;
#if BCPU(LITTLE_ENDIAN)
        memcpy(&value, m_storage.data(), storageSize);
#else
        memcpy(std::bit_cast<uint8_t*>(&value) + (sizeof(void*) - storageSize), m_storage.data(), storageSize);
#endif
        if (isAlignmentShiftProfitable)
            value <<= alignmentShiftSize;
        return std::bit_cast<T*>(value);
    }

    void set(T* passedValue)
    {
        uintptr_t value = std::bit_cast<uintptr_t>(passedValue);
        if (isAlignmentShiftProfitable)
            value >>= alignmentShiftSize;
#if BCPU(LITTLE_ENDIAN)
        memcpy(m_storage.data(), &value, storageSize);
#else
        memcpy(m_storage.data(), std::bit_cast<uint8_t*>(&value) + (sizeof(void*) - storageSize), storageSize);
#endif
    }

    void clear()
    {
        set(nullptr);
    }

    T* operator->() const { return get(); }
    T& operator*() const { return *get(); }
    bool operator!() const { return !get(); }

    // This conversion operator allows implicit conversion to bool but not to other integer types.
    typedef T* (PackedAlignedPtr::*UnspecifiedBoolType);
    operator UnspecifiedBoolType() const { return get() ? &PackedAlignedPtr::m_storage : nullptr; }
    explicit operator bool() const { return get(); }

    PackedAlignedPtr& operator=(T* value)
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

    void swap(PackedAlignedPtr& other)
    {
        m_storage.swap(other.m_storage);
    }

    template<typename Other, typename = std::enable_if_t<Other::isPackedType>>
    void swap(Other& other)
    {
        T* t1 = get();
        T* t2 = other.get();
        set(t2);
        other.set(t1);
    }

    void swap(T* t2)
    {
        T* t1 = get();
        std::swap(t1, t2);
        set(t1);
    }

private:
    std::array<uint8_t, storageSize> m_storage;
};

template<typename T>
class Packed<T*> : public PackedAlignedPtr<T, 1> {
public:
    using Base = PackedAlignedPtr<T, 1>;
    using Base::Base;
};

template<typename T>
using PackedPtr = Packed<T*>;

template<typename T>
struct PackedPtrTraits {
    template<typename U> using RebindTraits = PackedPtrTraits<U>;

    using StorageType = PackedPtr<T>;

    template<class U> static T* exchange(StorageType& ptr, U&& newValue) { return ptr.exchange(newValue); }

    template<typename Other> static void swap(PackedPtr<T>& a, Other& b) { a.swap(b); }

    static T* unwrap(const StorageType& ptr) { return ptr.get(); }
};

} // namespace bmalloc

#endif
