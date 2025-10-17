/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 27, 2022.
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

#include <limits.h>
#include <wtf/CryptographicallyRandomNumber.h>
#include <wtf/StdLibExtras.h>

namespace WTF {

// The code used to generate random numbers are inlined manually in JIT code.
// So it needs to stay in sync with the JIT one.
class WeakRandom final {
    WTF_MAKE_FAST_ALLOCATED;
public:
    WeakRandom(unsigned seed = cryptographicallyRandomNumber<unsigned>())
    {
        setSeed(seed);
    }

    void setSeed(unsigned seed)
    {
        m_seed = seed;

        // A zero seed would cause an infinite series of zeroes.
        if (!seed)
            seed = 1;

        m_low = seed;
        m_high = seed;
        advance();
    }

    unsigned seed() const { return m_seed; }

    double get()
    {
        uint64_t value = advance() & ((1ULL << 53) - 1);
        return value * (1.0 / (1ULL << 53));
    }

    unsigned getUint32()
    {
        return static_cast<unsigned>(advance());
    }

    unsigned getUint32(unsigned limit)
    {
        if (limit <= 1)
            return 0;
        uint64_t cutoff = (static_cast<uint64_t>(std::numeric_limits<unsigned>::max()) + 1) / limit * limit;
        for (;;) {
            uint64_t value = getUint32();
            if (value >= cutoff)
                continue;
            return value % limit;
        }
    }

    uint64_t getUint64()
    {
        return advance();
    }

    bool returnTrueWithProbability(double probability)
    {
        ASSERT(0.0 <= probability && probability <= 1.0);

        if (!probability)
            return false;

        double value = getUint32();
        if (value <= static_cast<double>(std::numeric_limits<unsigned>::max()) * probability)
            return true;
        return false;
    }

    static constexpr unsigned lowOffset() { return OBJECT_OFFSETOF(WeakRandom, m_low); }
    static constexpr unsigned highOffset() { return OBJECT_OFFSETOF(WeakRandom, m_high); }

    static constexpr uint64_t nextState(uint64_t x, uint64_t y)
    {
        x ^= x << 23;
        x ^= x >> 17;
        x ^= y ^ (y >> 26);
        return x;
    }

    static constexpr uint64_t generate(unsigned seed)
    {
        if (!seed)
            seed = 1;
        uint64_t low = seed;
        uint64_t high = seed;
        high = nextState(low, high);
        return low + high;
    }

private:
    uint64_t advance()
    {
        uint64_t x = m_low;
        uint64_t y = m_high;
        m_low = y;
        m_high = nextState(x, y);
        return m_high + m_low;
    }

    unsigned m_seed;
    uint64_t m_low;
    uint64_t m_high;
};

} // namespace WTF

using WTF::WeakRandom;
