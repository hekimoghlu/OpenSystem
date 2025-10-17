/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 23, 2025.
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
#include <wtf/Assertions.h>

namespace JSC {

class VM;

namespace DFG {

struct DoesGCCheck {
    enum class Special {
        Uninitialized,
        DFGOSRExit,
        FTLOSRExit,
        NumberOfSpecials
    };

    DoesGCCheck()
    {
        u.encoded = encode(true, Special::Uninitialized);
    }

    static uint64_t encode(bool expectDoesGC, unsigned nodeIndex, unsigned nodeOp)
    {
        Union un;
        un.nodeIndex = nodeIndex;
        un.other = (nodeOp << nodeOpShift) | bits(expectDoesGC);
        return un.encoded;
    }

    static uint64_t encode(bool expectDoesGC, Special special)
    {
        Union un;
        un.nodeIndex = 0;
        un.other = bits(special) << specialShift | isSpecialBit | bits(expectDoesGC);
        return un.encoded;
    }

    void set(bool expectDoesGC, unsigned nodeIndex, unsigned nodeOp)
    {
        u.encoded = encode(expectDoesGC, nodeIndex, nodeOp);
    }

    void set(bool expectDoesGC, Special special)
    {
        u.encoded = encode(expectDoesGC, special);
    }

    bool expectDoesGC() const { return u.other & expectDoesGCBit; }
    bool isSpecial() const { return u.other & isSpecialBit; }
    Special special() { return static_cast<Special>(u.other >> specialShift); }
    unsigned nodeOp() { return (u.other >> nodeOpShift) & nodeOpMask; }
    unsigned nodeIndex() { return u.nodeIndex; }

#if ENABLE(DFG_DOES_GC_VALIDATION)
    JS_EXPORT_PRIVATE void verifyCanGC(VM&);
#endif

private:
    template<typename T> static uint32_t bits(T value) { return static_cast<uint32_t>(value); }

    // The value cannot be both a Special and contain node information at the
    // time. Hence, the 2 can have separate encodings. The isSpecial bit
    // determines which encoding is in use.

    static constexpr unsigned expectDoesGCBit = 1 << 0;
    static constexpr unsigned isSpecialBit = 1 << 1;
    static constexpr unsigned commonBits = 2;
    static_assert((expectDoesGCBit | isSpecialBit) == (1 << commonBits) - 1);

    static constexpr unsigned specialShift = commonBits;

    static constexpr unsigned nodeOpBits = 9;
    static constexpr unsigned nodeOpMask = (1 << nodeOpBits) - 1;
    static constexpr unsigned nodeOpShift = commonBits;

public:
    union Union {
        struct {
            uint32_t other;
            uint32_t nodeIndex;
        };
        uint64_t encoded;
    } u;
};

} // namespace DFG

using DFG::DoesGCCheck;

} // namespace JSC
