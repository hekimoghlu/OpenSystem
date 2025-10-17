/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 16, 2023.
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

#include "DFGOSRExitBase.h"
#include "DFGVariableEventStream.h"
#include "GPRInfo.h"
#include "MacroAssembler.h"
#include "MethodOfGettingAValueProfile.h"
#include "Operands.h"
#include "ValueRecovery.h"
#include <wtf/RefPtr.h>
#include <wtf/TZoneMalloc.h>

namespace JSC {

class ArrayProfile;
class CCallHelpers;

namespace Probe {
class Context;
} // namespace Probe

namespace Profiler {
class OSRExit;
} // namespace Profiler

namespace DFG {

class BasicBlock;
class SpeculativeJIT;
struct Node;

// This enum describes the types of additional recovery that
// may need be performed should a speculation check fail.
enum SpeculationRecoveryType : uint8_t {
    SpeculativeAdd,
    SpeculativeAddSelf,
    SpeculativeAddImmediate,
    BooleanSpeculationCheck
};

// === SpeculationRecovery ===
//
// This class provides additional information that may be associated with a
// speculation check - for example 
class SpeculationRecovery {
public:
    SpeculationRecovery(SpeculationRecoveryType type, GPRReg dest, GPRReg src)
        : m_src(src)
        , m_dest(dest)
        , m_type(type)
    {
        ASSERT(m_type == SpeculativeAdd || m_type == SpeculativeAddSelf || m_type == BooleanSpeculationCheck);
    }

    SpeculationRecovery(SpeculationRecoveryType type, GPRReg dest, int32_t immediate)
        : m_immediate(immediate)
        , m_dest(dest)
        , m_type(type)
    {
        ASSERT(m_type == SpeculativeAddImmediate);
    }

    SpeculationRecoveryType type() { return m_type; }
    GPRReg dest() { return m_dest; }
    GPRReg src() { return m_src; }
    int32_t immediate() { return m_immediate; }

private:
    // different recovery types may required different additional information here.
    union {
        GPRReg m_src;
        int32_t m_immediate;
    };
    GPRReg m_dest;

    // Indicates the type of additional recovery to be performed.
    SpeculationRecoveryType m_type;
};

JSC_DECLARE_NOEXCEPT_JIT_OPERATION(operationCompileOSRExit, void, (CallFrame*, void*));
JSC_DECLARE_NOEXCEPT_JIT_OPERATION(operationDebugPrintSpeculationFailure, void, (Probe::Context&));
JSC_DECLARE_NOEXCEPT_JIT_OPERATION(operationMaterializeOSRExitSideState, void, (VM*, const OSRExitBase*, EncodedJSValue*));

// === OSRExit ===
//
// This structure describes how to exit the speculative path by
// going into baseline code.
struct OSRExit : public OSRExitBase {
    OSRExit(ExitKind, JSValueSource, MethodOfGettingAValueProfile, SpeculativeJIT*, unsigned streamIndex, unsigned recoveryIndex = UINT_MAX);

    CodeLocationLabel<JSInternalPtrTag> m_patchableJumpLocation;

    JSValueSource m_jsValueSource;
    MethodOfGettingAValueProfile m_valueProfile;
    
    unsigned m_recoveryIndex;

    CodeLocationJump<JSInternalPtrTag> codeLocationForRepatch() const { return CodeLocationJump<JSInternalPtrTag>(m_patchableJumpLocation); }

    unsigned m_streamIndex;
    void considerAddingAsFrequentExitSite(CodeBlock* profiledCodeBlock)
    {
        OSRExitBase::considerAddingAsFrequentExitSite(profiledCodeBlock, ExitFromDFG);
    }

    static void compileExit(CCallHelpers&, VM&, const OSRExit&, const Operands<ValueRecovery>&, SpeculationRecovery*, uint32_t osrExitIndex);

private:
    static void emitRestoreArguments(CCallHelpers&, VM&, const Operands<ValueRecovery>&);
};

struct SpeculationFailureDebugInfo {
    WTF_MAKE_STRUCT_TZONE_ALLOCATED(SpeculationFailureDebugInfo);
    CodeBlock* codeBlock;
    ExitKind kind;
    uint32_t exitIndex;
    BytecodeIndex bytecodeIndex;
};

} } // namespace JSC::DFG

#endif // ENABLE(DFG_JIT)
