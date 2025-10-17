/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 11, 2025.
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

#include <wtf/StdLibExtras.h>

namespace JSC {

template <typename Bits = uintptr_t>
class TinyBloomFilter {
public:
    TinyBloomFilter() = default;
    TinyBloomFilter(Bits);

    void add(Bits);
    void add(TinyBloomFilter&);
    bool ruleOut(Bits) const; // True for 0.
    void reset();
    Bits bits() const { return m_bits; }

    static constexpr ptrdiff_t offsetOfBits() { return OBJECT_OFFSETOF(TinyBloomFilter, m_bits); }

private:
    Bits m_bits { 0 };
};

template <typename Bits>
inline TinyBloomFilter<Bits>::TinyBloomFilter(Bits bits)
    : m_bits(bits)
{
}

template <typename Bits>
inline void TinyBloomFilter<Bits>::add(Bits bits)
{
    m_bits |= bits;
}

template <typename Bits>
inline void TinyBloomFilter<Bits>::add(TinyBloomFilter& other)
{
    m_bits |= other.m_bits;
}

template <typename Bits>
inline bool TinyBloomFilter<Bits>::ruleOut(Bits bits) const
{
    if (!bits)
        return true;

    if ((bits & m_bits) != bits)
        return true;

    return false;
}

template <typename Bits>
inline void TinyBloomFilter<Bits>::reset()
{
    m_bits = 0;
}

} // namespace JSC
