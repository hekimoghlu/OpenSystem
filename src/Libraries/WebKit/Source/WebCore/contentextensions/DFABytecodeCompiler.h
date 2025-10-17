/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 30, 2022.
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

#if ENABLE(CONTENT_EXTENSIONS)

#include "DFABytecode.h"
#include <wtf/Vector.h>

namespace WebCore {

namespace ContentExtensions {

struct DFA;
class DFANode;

class WEBCORE_EXPORT DFABytecodeCompiler {
public:
    DFABytecodeCompiler(const DFA& dfa, Vector<DFABytecode>& bytecode)
        : m_bytecode(bytecode)
        , m_dfa(dfa)
    {
    }
    
    void compile();

private:
    struct Range {
        Range(uint8_t min, uint8_t max, uint32_t destination, bool caseSensitive)
            : min(min)
            , max(max)
            , destination(destination)
            , caseSensitive(caseSensitive)
        {
        }
        uint8_t min;
        uint8_t max;
        uint32_t destination;
        bool caseSensitive;
    };
    struct JumpTable {
        ~JumpTable()
        {
            ASSERT(min + destinations.size() == static_cast<size_t>(max + 1));
            ASSERT(min == max || destinations.size() > 1);
        }

        uint8_t min { 0 };
        uint8_t max { 0 };
        bool caseSensitive { true };
        Vector<uint32_t> destinations;
    };
    struct Transitions {
        Vector<JumpTable> jumpTables;
        Vector<Range> ranges;
        bool useFallbackTransition { false };
        uint32_t fallbackTransitionTarget { std::numeric_limits<uint32_t>::max() };
    };
    JumpTable extractJumpTable(Vector<Range>&, unsigned first, unsigned last);
    Transitions transitions(const DFANode&);
    
    unsigned compiledNodeMaxBytecodeSize(uint32_t index);
    void compileNode(uint32_t index, bool root);
    unsigned nodeTransitionsMaxBytecodeSize(const DFANode&);
    void compileNodeTransitions(uint32_t nodeIndex);
    unsigned checkForJumpTableMaxBytecodeSize(const JumpTable&);
    unsigned checkForRangeMaxBytecodeSize(const Range&);
    void compileJumpTable(uint32_t nodeIndex, const JumpTable&);
    void compileCheckForRange(uint32_t nodeIndex, const Range&);
    int32_t longestPossibleJump(uint32_t jumpLocation, uint32_t sourceNodeIndex, uint32_t destinationNodeIndex);

    void emitAppendAction(uint64_t);
    void emitJump(uint32_t sourceNodeIndex, uint32_t destinationNodeIndex);
    void emitCheckValue(uint8_t value, uint32_t sourceNodeIndex, uint32_t destinationNodeIndex, bool caseSensitive);
    void emitCheckValueRange(uint8_t lowValue, uint8_t highValue, uint32_t sourceNodeIndex, uint32_t destinationNodeIndex, bool caseSensitive);
    void emitTerminate();

    Vector<DFABytecode>& m_bytecode;
    const DFA& m_dfa;
    
    Vector<uint32_t> m_maxNodeStartOffsets;
    Vector<uint32_t> m_nodeStartOffsets;
    
    struct LinkRecord {
        DFABytecodeJumpSize jumpSize;
        int32_t longestPossibleJump;
        uint32_t instructionLocation;
        uint32_t jumpLocation;
        uint32_t destinationNodeIndex;
    };
    Vector<LinkRecord> m_linkRecords;
};

} // namespace ContentExtensions
} // namespace WebCore

#endif // ENABLE(CONTENT_EXTENSIONS)
