/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 4, 2022.
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

#if ENABLE(B3_JIT)

#include "B3Origin.h"
#include "MacroAssembler.h"
#include <wtf/Vector.h>

namespace JSC { namespace B3 {

class PCToOriginMap {
    WTF_MAKE_NONCOPYABLE(PCToOriginMap);
public:
    PCToOriginMap()
    { }

    PCToOriginMap(PCToOriginMap&& other)
        : m_ranges(WTFMove(other.m_ranges))
    { }

    struct OriginRange {
        MacroAssembler::Label label;
        Origin origin;
    };

    void appendItem(MacroAssembler::Label label, Origin origin)
    {
        if (m_ranges.size()) {
            if (m_ranges.last().label == label)
                return;
        }

        m_ranges.append(OriginRange{label, origin});
    }

    const Vector<OriginRange>& ranges() const  { return m_ranges; }

private:
    Vector<OriginRange, 0> m_ranges;
};

} } // namespace JSC::B3

#endif // ENABLE(B3_JIT)
