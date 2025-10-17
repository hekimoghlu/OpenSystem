/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 7, 2025.
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
#include "config.h"
#include "FTLLink.h"

#if ENABLE(FTL_JIT)

#include "CCallHelpers.h"
#include "CodeBlockWithJITType.h"
#include "FTLJITCode.h"
#include "JITOperations.h"
#include "JITThunks.h"
#include "LinkBuffer.h"
#include "ProfilerCompilation.h"
#include "ThunkGenerators.h"

namespace JSC { namespace FTL {

void link(State& state)
{
    using namespace DFG;
    Graph& graph = state.graph;
    CodeBlock* codeBlock = graph.m_codeBlock;

    state.jitCode->common.requiredRegisterCountForExit = graph.requiredRegisterCountForExit();

    if (!graph.m_plan.inlineCallFrames()->isEmpty())
        state.jitCode->common.inlineCallFrames = graph.m_plan.inlineCallFrames();
    if (!graph.m_stringSearchTable8.isEmpty()) {
        FixedVector<std::unique_ptr<BoyerMooreHorspoolTable<uint8_t>>> tables(graph.m_stringSearchTable8.size());
        unsigned index = 0;
        for (auto& entry : graph.m_stringSearchTable8)
            tables[index++] = WTFMove(entry.value);
        state.jitCode->common.m_stringSearchTable8 = WTFMove(tables);
    }

    graph.registerFrozenValues();

    {
        bool dumpDisassembly = shouldDumpDisassembly() || Options::asyncDisassembly();

        MacroAssemblerCodeRef<JSEntryPtrTag> b3CodeRef =
            FINALIZE_CODE_IF(dumpDisassembly, *state.b3CodeLinkBuffer, JSEntryPtrTag, nullptr,
                "FTL B3 code for %s", toCString(CodeBlockWithJITType(codeBlock, JITType::FTLJIT)).data());

        state.jitCode->initializeB3Code(b3CodeRef);
        state.jitCode->common.m_jumpReplacements = WTFMove(state.jumpReplacements);
    }

    state.finalizer->m_codeSize = state.b3CodeLinkBuffer->size();
    state.finalizer->m_jitCode = state.jitCode;
}

} } // namespace JSC::FTL

#endif // ENABLE(FTL_JIT)

