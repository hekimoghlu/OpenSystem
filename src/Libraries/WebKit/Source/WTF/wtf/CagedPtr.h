/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 1, 2025.
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

#include <wtf/Gigacage.h>
#include <wtf/MathExtras.h>
#include <wtf/PtrTag.h>
#include <wtf/RawPtrTraits.h>

#include <climits>

#if OS(DARWIN)
#include <mach/vm_param.h>
#endif

namespace WTF {

template<Gigacage::Kind passedKind, typename T, typename PtrTraits = RawPtrTraits<T>>
class CagedPtr {
public:
    static constexpr Gigacage::Kind kind = passedKind;
    static constexpr unsigned numberOfPointerBits = sizeof(T*) * CHAR_BIT;
    static constexpr unsigned maxNumberOfAllowedPACBits = numberOfPointerBits - OS_CONSTANT(EFFECTIVE_ADDRESS_WIDTH);
    static constexpr uintptr_t nonPACBitsMask = (1ull << (numberOfPointerBits - maxNumberOfAllowedPACBits)) - 1;

    CagedPtr() : CagedPtr(nullptr) { }
    CagedPtr(std::nullptr_t)
        : m_ptr(nullptr)
    { }

    CagedPtr(T* ptr)
        : m_ptr(ptr)
    { }

    T* get() const
    {
        ASSERT(m_ptr);
        T* ptr = PtrTraits::unwrap(m_ptr);
        return Gigacage::caged(kind, ptr);
    }

    T* getMayBeNull() const
    {
        T* ptr = PtrTraits::unwrap(m_ptr);
        if (!ptr)
            return nullptr;
        return Gigacage::caged(kind, ptr);
    }

    T* getUnsafe() const
    {
        T* ptr = PtrTraits::unwrap(m_ptr);
        return Gigacage::cagedMayBeNull(kind, ptr);
    }

    // We need the template here so that the type of U is deduced at usage time rather than class time. U should always be T.
    template<typename U = T>
    typename std::enable_if<!std::is_same<void, U>::value, T>::type&
    /* T& */ at(size_t index) const { return get()[index]; }

    CagedPtr(CagedPtr& other)
        : m_ptr(other.m_ptr)
    {
    }

    CagedPtr& operator=(const CagedPtr& ptr)
    {
        m_ptr = ptr.m_ptr;
        return *this;
    }

    CagedPtr(CagedPtr&& other)
        : m_ptr(PtrTraits::exchange(other.m_ptr, nullptr))
    {
    }

    CagedPtr& operator=(CagedPtr&& ptr)
    {
        m_ptr = PtrTraits::exchange(ptr.m_ptr, nullptr);
        return *this;
    }

    bool operator==(const CagedPtr& other) const
    {
        bool result = m_ptr == other.m_ptr;
        ASSERT(result == (getUnsafe() == other.getUnsafe()));
        return result;
    }
    
    explicit operator bool() const
    {
        return getUnsafe() != nullptr;
    }

    T* rawBits() const
    {
        return std::bit_cast<T*>(m_ptr);
    }
    
protected:
    typename PtrTraits::StorageType m_ptr;
};

} // namespace WTF

using WTF::CagedPtr;

