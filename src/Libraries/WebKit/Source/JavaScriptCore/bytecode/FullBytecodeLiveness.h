/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 21, 2025.
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

#include "BytecodeLivenessAnalysis.h"
#include "Operands.h"
#include <wtf/FastBitVector.h>
#include <wtf/FixedVector.h>
#include <wtf/TZoneMalloc.h>

namespace JSC {

class BytecodeLivenessAnalysis;
class CodeBlock;

// Note: Full bytecode liveness does not track any information about the liveness of temps.
// If you want tmp liveness for a checkpoint ask tmpLivenessForCheckpoint.
class FullBytecodeLiveness {
    WTF_MAKE_TZONE_ALLOCATED(FullBytecodeLiveness);
public:
    explicit FullBytecodeLiveness(size_t size)
        : m_usesBefore(size)
        , m_usesAfter(size)
    { }

    const FastBitVector& getLiveness(BytecodeIndex bytecodeIndex, LivenessCalculationPoint point) const
    {
        // We don't have to worry about overflowing into the next bytecodeoffset in our vectors because we
        // static assert that bytecode length is greater than the number of checkpoints in BytecodeStructs.h
        switch (point) {
        case LivenessCalculationPoint::BeforeUse:
            return m_usesBefore[toIndex(bytecodeIndex)];
        case LivenessCalculationPoint::AfterUse:
            return m_usesAfter[toIndex(bytecodeIndex)];
        }
        RELEASE_ASSERT_NOT_REACHED();
    }
    
    bool virtualRegisterIsLive(VirtualRegister reg, BytecodeIndex bytecodeIndex, LivenessCalculationPoint point) const
    {
        return virtualRegisterIsAlwaysLive(reg) || virtualRegisterThatIsNotAlwaysLiveIsLive(getLiveness(bytecodeIndex, point), reg);
    }
    
private:
    friend class BytecodeLivenessAnalysis;
    
    static size_t toIndex(BytecodeIndex bytecodeIndex) { return bytecodeIndex.offset() + bytecodeIndex.checkpoint(); }

    // FIXME: Use FastBitVector's view mechanism to make them compact.
    // https://bugs.webkit.org/show_bug.cgi?id=204427
    FixedVector<FastBitVector> m_usesBefore;
    FixedVector<FastBitVector> m_usesAfter;
};

} // namespace JSC
