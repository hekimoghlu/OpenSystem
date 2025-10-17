/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 6, 2024.
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

#include "BytecodeBasicBlock.h"
#include "BytecodeGraph.h"
#include "CodeBlock.h"
#include <wtf/BitSet.h>
#include <wtf/FastBitVector.h>

namespace JSC {

class BytecodeKills;
class FullBytecodeLiveness;

// We model our bytecode effects like the following and insert the liveness calculation points.
//
// <- BeforeUse
//     Use
// <- AfterUse
//     Use by exception handlers
//     Def
enum class LivenessCalculationPoint : uint8_t {
    BeforeUse,
    AfterUse,
};

class BytecodeLivenessPropagation {
public:
    template<typename CodeBlockType, typename UseFunctor>
    static void stepOverBytecodeIndexUse(CodeBlockType*, const JSInstructionStream&, BytecodeGraph&, BytecodeIndex, const UseFunctor&);
    template<typename CodeBlockType, typename UseFunctor>
    static void stepOverBytecodeIndexUseInExceptionHandler(CodeBlockType*, const JSInstructionStream&, BytecodeGraph&, BytecodeIndex, const UseFunctor&);
    template<typename CodeBlockType, typename DefFunctor>
    static void stepOverBytecodeIndexDef(CodeBlockType*, const JSInstructionStream&, BytecodeGraph&, BytecodeIndex, const DefFunctor&);

    template<typename CodeBlockType, typename UseFunctor, typename DefFunctor>
    static void stepOverBytecodeIndex(CodeBlockType*, const JSInstructionStream&, BytecodeGraph&, BytecodeIndex, const UseFunctor&, const DefFunctor&);

    template<typename CodeBlockType>
    static void stepOverInstruction(CodeBlockType*, const JSInstructionStream&, BytecodeGraph&, BytecodeIndex, FastBitVector& out);

    template<typename CodeBlockType, typename Instructions>
    static bool computeLocalLivenessForInstruction(CodeBlockType*, const Instructions&, BytecodeGraph&, JSBytecodeBasicBlock&, BytecodeIndex, FastBitVector& result);

    template<typename CodeBlockType, typename Instructions>
    static bool computeLocalLivenessForBlock(CodeBlockType*, const Instructions&, BytecodeGraph&, JSBytecodeBasicBlock&);

    template<typename CodeBlockType, typename Instructions>
    static FastBitVector getLivenessInfoAtInstruction(CodeBlockType*, const Instructions&, BytecodeGraph&, BytecodeIndex);

    template<typename CodeBlockType, typename Instructions>
    static void runLivenessFixpoint(CodeBlockType*, const Instructions&, BytecodeGraph&);
};

class BytecodeLivenessAnalysis : private BytecodeLivenessPropagation {
    WTF_MAKE_TZONE_ALLOCATED(BytecodeLivenessAnalysis);
    WTF_MAKE_NONCOPYABLE(BytecodeLivenessAnalysis);
public:
    friend class BytecodeLivenessPropagation;
    BytecodeLivenessAnalysis(CodeBlock*);
    
    FastBitVector getLivenessInfoAtInstruction(CodeBlock* codeBlock, BytecodeIndex bytecodeIndex) { return BytecodeLivenessPropagation::getLivenessInfoAtInstruction(codeBlock, codeBlock->instructions(), m_graph, bytecodeIndex); }
    
    std::unique_ptr<FullBytecodeLiveness> computeFullLiveness(CodeBlock*);

    BytecodeGraph& graph() { return m_graph; }

private:
    void dumpResults(CodeBlock*);

    BytecodeGraph m_graph;
};

WTF::BitSet<maxNumCheckpointTmps> tmpLivenessForCheckpoint(const CodeBlock&, BytecodeIndex);

inline bool operandIsAlwaysLive(int operand);
inline bool operandThatIsNotAlwaysLiveIsLive(const FastBitVector& out, int operand);
inline bool operandIsLive(const FastBitVector& out, int operand);
inline bool isValidRegisterForLiveness(int operand);

} // namespace JSC
