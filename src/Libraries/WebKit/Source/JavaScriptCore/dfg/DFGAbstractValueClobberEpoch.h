/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 8, 2025.
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

#if ENABLE(DFG_JIT)

#include "DFGEpoch.h"
#include "DFGStructureClobberState.h"
#include <wtf/PrintStream.h>

namespace JSC { namespace DFG {

class AbstractValueClobberEpoch {
public:
    AbstractValueClobberEpoch()
    {
    }
    
    static AbstractValueClobberEpoch first(StructureClobberState clobberState)
    {
        AbstractValueClobberEpoch result;
        result.m_value = Epoch::first().toUnsigned() << epochShift;
        switch (clobberState) {
        case StructuresAreWatched:
            result.m_value |= watchedFlag;
            break;
        case StructuresAreClobbered:
            break;
        }
        return result;
    }
    
    void clobber()
    {
        m_value += step;
        m_value &= ~watchedFlag;
    }
    
    void observeInvalidationPoint()
    {
        m_value |= watchedFlag;
    }
    
    friend bool operator==(const AbstractValueClobberEpoch&, const AbstractValueClobberEpoch&) = default;
    
    StructureClobberState structureClobberState() const
    {
        return m_value & watchedFlag ? StructuresAreWatched : StructuresAreClobbered;
    }
    
    Epoch clobberEpoch() const
    {
        return Epoch::fromUnsigned(m_value >> epochShift);
    }
    
    void dump(PrintStream&) const;
    
private:
    static constexpr unsigned step = 2;
    static constexpr unsigned watchedFlag = 1;
    static constexpr unsigned epochShift = 1;
    
    unsigned m_value { 0 };
};

} } // namespace JSC::DFG

#endif // ENABLE(DFG_JIT)
