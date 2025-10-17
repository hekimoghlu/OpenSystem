/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 18, 2025.
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

#include <wtf/Assertions.h>
#include <wtf/HashTraits.h>
#include <wtf/UniqueArray.h>

namespace WTF {

class StackShot {
    WTF_MAKE_FAST_ALLOCATED;
public:
    StackShot() { }
    
    ALWAYS_INLINE StackShot(size_t size)
        : m_size(size)
    {
        if (size) {
            m_array = makeUniqueArray<void*>(size);
            int intSize = size;
            WTFGetBacktrace(m_array.get(), &intSize);
            RELEASE_ASSERT(static_cast<size_t>(intSize) <= size);
            m_size = intSize;
            if (!m_size)
                m_array = nullptr;
        }
    }
    
    StackShot(WTF::HashTableDeletedValueType)
        : m_array(deletedValueArray())
        , m_size(0)
    {
    }
    
    StackShot& operator=(const StackShot& other)
    {
        auto newArray = makeUniqueArray<void*>(other.m_size);
        for (size_t i = other.m_size; i--;)
            newArray[i] = other.m_array[i];
        m_size = other.m_size;
        m_array = WTFMove(newArray);
        return *this;
    }
    
    StackShot(const StackShot& other)
    {
        *this = other;
    }
    
    void** array() const LIFETIME_BOUND { return m_array.get(); }
    size_t size() const { return m_size; }

    std::span<void*> span() const LIFETIME_BOUND { return unsafeMakeSpan(m_array.get(), m_size); }
    
    explicit operator bool() const { return !!m_array; }
    
    bool operator==(const StackShot& other) const
    {
        if (m_size != other.m_size)
            return false;
        
        for (size_t i = m_size; i--;) {
            if (m_array[i] != other.m_array[i])
                return false;
        }
        
        return true;
    }
    
    unsigned hash() const
    {
        unsigned result = m_size;
        
        for (size_t i = m_size; i--;)
            result ^= PtrHash<void*>::hash(m_array[i]);
        
        return result;
    }
    
    bool isHashTableDeletedValue() const
    {
        return !m_size && m_array.get() == deletedValueArray();
    }
    
    // Make Spectrum<> happy.
    bool operator>(const StackShot&) const { return false; }
    
private:
    static void** deletedValueArray() { return std::bit_cast<void**>(static_cast<uintptr_t>(1)); }

    UniqueArray<void*> m_array;
    size_t m_size { 0 };
};

struct StackShotHash {
    static unsigned hash(const StackShot& shot) { return shot.hash(); }
    static bool equal(const StackShot& a, const StackShot& b) { return a == b; }
    static constexpr bool safeToCompareToEmptyOrDeleted = false;
};

template<typename T> struct DefaultHash;
template<> struct DefaultHash<StackShot> : StackShotHash { };

template<> struct HashTraits<StackShot> : SimpleClassHashTraits<StackShot> { };

} // namespace WTF

