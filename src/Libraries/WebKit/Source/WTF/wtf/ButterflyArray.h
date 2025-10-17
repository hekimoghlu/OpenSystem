/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 19, 2024.
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

#include <type_traits>
#include <wtf/StdLibExtras.h>
#include <wtf/Vector.h>

WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN

namespace WTF {

// ButterflyArray offers the feature trailing and leading array in the derived class.
// We can allocate a memory like the following layout.
//
//     [ Leading Array ][  DerivedClass  ][ Trailing Array ]
template<typename Derived, typename LeadingType, typename TrailingType>
class ButterflyArray {
    WTF_MAKE_NONCOPYABLE(ButterflyArray);
    friend class JSC::LLIntOffsetsExtractor;
protected:
    explicit ButterflyArray(unsigned leadingSize, unsigned trailingSize)
        : m_leadingSize(leadingSize)
        , m_trailingSize(trailingSize)
    {
        static_assert(std::is_final_v<Derived>);
        auto leadingSpan = this->leadingSpan();
        VectorTypeOperations<LeadingType>::initializeIfNonPOD(leadingSpan.data(), leadingSpan.data() + leadingSpan.size());
        auto trailingSpan = this->trailingSpan();
        VectorTypeOperations<TrailingType>::initializeIfNonPOD(trailingSpan.data(), trailingSpan.data() + trailingSpan.size());
    }

    template<typename... Args>
    static Derived* createImpl(unsigned leadingSize, unsigned trailingSize, Args&&... args)
    {
        uint8_t* memory = std::bit_cast<uint8_t*>(fastMalloc(allocationSize(leadingSize, trailingSize)));
        return new (NotNull, memory + memoryOffsetForDerived(leadingSize)) Derived(leadingSize, trailingSize, std::forward<Args>(args)...);
    }

public:
    static constexpr size_t allocationSize(unsigned leadingSize, unsigned trailingSize)
    {
        return memoryOffsetForDerived(leadingSize) + offsetOfTrailingData() + trailingSize * sizeof(TrailingType);
    }

    static constexpr ptrdiff_t offsetOfLeadingSize() { return OBJECT_OFFSETOF(Derived, m_leadingSize); }
    static constexpr ptrdiff_t offsetOfTrailingSize() { return OBJECT_OFFSETOF(Derived, m_trailingSize); }
    static constexpr ptrdiff_t offsetOfTrailingData()
    {
        return WTF::roundUpToMultipleOf<alignof(TrailingType)>(sizeof(Derived));
    }

    static constexpr ptrdiff_t memoryOffsetForDerived(unsigned leadingSize)
    {
        return WTF::roundUpToMultipleOf<alignof(Derived)>(sizeof(LeadingType) * leadingSize);
    }

    std::span<LeadingType> leadingSpan()
    {
        return std::span { leadingData(), m_leadingSize };
    }

    std::span<const LeadingType> leadingSpan() const
    {
        return std::span { leadingData(), m_leadingSize };
    }

    std::span<TrailingType> trailingSpan()
    {
        return std::span { trailingData(), m_trailingSize };
    }

    std::span<const TrailingType> trailingSpan() const
    {
        return std::span { trailingData(), m_trailingSize };
    }

    void operator delete(ButterflyArray* base, std::destroying_delete_t)
    {
        unsigned leadingSize = base->m_leadingSize;
        std::destroy_at(static_cast<Derived*>(base));
        fastFree(std::bit_cast<uint8_t*>(static_cast<Derived*>(base)) - memoryOffsetForDerived(leadingSize));
    }

    ~ButterflyArray()
    {
        auto leadingSpan = this->leadingSpan();
        VectorTypeOperations<LeadingType>::destruct(leadingSpan.data(), leadingSpan.data() + leadingSpan.size());
        auto trailingSpan = this->trailingSpan();
        VectorTypeOperations<TrailingType>::destruct(trailingSpan.data(), trailingSpan.data() + trailingSpan.size());
    }

protected:
    LeadingType* leadingData()
    {
        return std::bit_cast<LeadingType*>(static_cast<Derived*>(this)) - m_leadingSize;
    }

    const LeadingType leadingData() const
    {
        return std::bit_cast<const LeadingType*>(static_cast<const Derived*>(this)) - m_leadingSize;
    }

    TrailingType* trailingData()
    {
        return std::bit_cast<TrailingType*>(std::bit_cast<uint8_t*>(static_cast<Derived*>(this)) + offsetOfTrailingData());
    }

    const TrailingType* trailingData() const
    {
        return std::bit_cast<const TrailingType*>(std::bit_cast<const uint8_t*>(static_cast<const Derived*>(this)) + offsetOfTrailingData());
    }

    unsigned m_leadingSize { 0 };
    unsigned m_trailingSize { 0 };
};

} // namespace WTF

using WTF::ButterflyArray;

WTF_ALLOW_UNSAFE_BUFFER_USAGE_END
