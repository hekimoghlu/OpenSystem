/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 15, 2022.
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

#include <wtf/HashMap.h>

namespace JSC {
namespace Wasm {

enum class BranchHint : uint8_t {
    Unlikely,
    Likely,
    Invalid,
};

class BranchHintMap {

public:

    void add(uint32_t branchOffset, BranchHint hint)
    {
        m_map.add(branchOffset, hint);
    }

    BranchHint getBranchHint(uint32_t branchOffset) const
    {
        auto it = m_map.find(branchOffset);
        if (it == m_map.end())
            return BranchHint::Invalid;
        return it->value;
    }

    bool isValidKey(uint32_t branchOffset) const
    {
        return m_map.isValidKey(branchOffset);
    }

private:

    UncheckedKeyHashMap<uint32_t, BranchHint, IntHash<uint32_t>, WTF::UnsignedWithZeroKeyHashTraits<uint32_t>> m_map;

};

inline constexpr bool isValidBranchHint(BranchHint hint)
{
    switch (hint) {
    case BranchHint::Likely:
    case BranchHint::Unlikely:
        return true;
    default:
        return false;
    }
    return false;
}

}
} // namespace JSC::Wasm
