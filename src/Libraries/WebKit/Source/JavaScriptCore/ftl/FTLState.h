/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 11, 2025.
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

#if ENABLE(FTL_JIT)

#include "B3Procedure.h"
#include "DFGCommon.h"
#include "DFGGraph.h"
#include "DFGJumpReplacement.h"
#include "FTLAbbreviatedTypes.h"
#include "FTLJITCode.h"
#include "FTLJITFinalizer.h"
#include <wtf/Box.h>
#include <wtf/Noncopyable.h>

namespace JSC {

namespace B3 {
class PatchpointValue;
namespace Air {
class StackSlot;
} // namespace Air
} // namespace B3

namespace FTL {

class PatchpointExceptionHandle;

inline bool verboseCompilationEnabled()
{
    return DFG::verboseCompilationEnabled(JITCompilationMode::FTL);
}

inline bool shouldDumpDisassembly()
{
    return DFG::shouldDumpDisassembly(JITCompilationMode::FTL);
}

class State {
    WTF_MAKE_NONCOPYABLE(State);
    
public:
    State(DFG::Graph& graph);
    ~State();

    VM& vm() { return graph.m_vm; }

    void dumpDisassembly(PrintStream&, LinkBuffer&, const ScopedLambda<void(DFG::Node*)>& perDFGNodeCallback = scopedLambda<void(DFG::Node*)>([] (DFG::Node*) { }));

    StructureStubInfo* addStructureStubInfo();
    OptimizingCallLinkInfo* addCallLinkInfo(CodeOrigin);

    // None of these things is owned by State. It is the responsibility of
    // FTL phases to properly manage the lifecycle of the module and function.
    DFG::Graph& graph;
    std::unique_ptr<B3::Procedure> proc;
    bool allocationFailed { false }; // Throw out the compilation once B3 returns.
    RefPtr<FTL::JITCode> jitCode;
    JITFinalizer* finalizer;
    std::unique_ptr<LinkBuffer> b3CodeLinkBuffer;
    // Top-level exception handler. Jump here if you know that you have to genericUnwind() and there
    // are no applicable catch blocks anywhere in the Graph.
    RefPtr<PatchpointExceptionHandle> defaultExceptionHandle;
    Box<CCallHelpers::Label> exceptionHandler { Box<CCallHelpers::Label>::create() };
    B3::Air::StackSlot* capturedValue { nullptr };
    Vector<DFG::JumpReplacement> jumpReplacements;
};

} } // namespace JSC::FTL

#endif // ENABLE(FTL_JIT)
