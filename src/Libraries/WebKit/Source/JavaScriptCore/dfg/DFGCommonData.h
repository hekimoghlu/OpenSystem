/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 18, 2024.
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

#include "BaselineJITCode.h"
#include "CallLinkInfo.h"
#include "CodeBlockJettisoningWatchpoint.h"
#include "DFGAdaptiveInferredPropertyValueWatchpoint.h"
#include "DFGAdaptiveStructureWatchpoint.h"
#include "DFGCodeOriginPool.h"
#include "DFGJumpReplacement.h"
#include "DFGOSREntry.h"
#include "InlineCallFrameSet.h"
#include "JITMathIC.h"
#include "JSCast.h"
#include "PCToCodeOriginMap.h"
#include "ProfilerCompilation.h"
#include "RecordedStatuses.h"
#include "StructureStubInfo.h"
#include "YarrJIT.h"
#include <wtf/Bag.h>
#include <wtf/Noncopyable.h>
#include <wtf/text/StringSearch.h>

namespace JSC {

class CodeBlock;
class Identifier;
class TrackedReferences;

namespace DFG {

struct Node;
class Plan;

// CommonData holds the set of data that both DFG and FTL code blocks need to know
// about themselves.

struct WeakReferenceTransition {
    WeakReferenceTransition() { }
    
    WeakReferenceTransition(VM& vm, JSCell* owner, JSCell* codeOrigin, JSCell* from, JSCell* to)
        : m_from(vm, owner, from)
        , m_to(vm, owner, to)
    {
        if (!!codeOrigin)
            m_codeOrigin.set(vm, owner, codeOrigin);
    }
    
    WriteBarrier<JSCell> m_codeOrigin;
    WriteBarrier<JSCell> m_from;
    WriteBarrier<JSCell> m_to;
};
        
class CommonData : public MathICHolder {
    WTF_MAKE_NONCOPYABLE(CommonData);
public:
    CommonData(bool isUnlinked)
        : codeOrigins(CodeOriginPool::create())
        , m_isUnlinked(isUnlinked)
    { }
    ~CommonData();
    
    void shrinkToFit();
    
    bool invalidateLinkedCode(); // Returns true if we did invalidate, or false if the code block was already invalidated.
    bool hasInstalledVMTrapsBreakpoints() const { return m_isStillValid && m_hasVMTrapsBreakpointsInstalled; }
    void installVMTrapBreakpoints(CodeBlock* owner);

    bool isUnlinked() const { return m_isUnlinked; }
    bool isStillValid() const { return m_isStillValid; }

    CatchEntrypointData* catchOSREntryDataForBytecodeIndex(BytecodeIndex bytecodeIndex)
    {
        return tryBinarySearch<CatchEntrypointData, BytecodeIndex>(
            m_catchEntrypoints, m_catchEntrypoints.size(), bytecodeIndex,
            [] (const CatchEntrypointData* item) { return item->bytecodeIndex; });
    }

    void finalizeCatchEntrypoints(Vector<CatchEntrypointData>&&);

    unsigned requiredRegisterCountForExecutionAndExit() const
    {
        return std::max(frameRegisterCount, requiredRegisterCountForExit);
    }
    
    void validateReferences(const TrackedReferences&);

    static constexpr ptrdiff_t frameRegisterCountOffset() { return OBJECT_OFFSETOF(CommonData, frameRegisterCount); }
    
    void clearWatchpoints();

    RefPtr<InlineCallFrameSet> inlineCallFrames;
    Ref<CodeOriginPool> codeOrigins;
    
    FixedVector<Identifier> m_dfgIdentifiers;
    FixedVector<WeakReferenceTransition> m_transitions;
    FixedVector<WriteBarrier<JSCell>> m_weakReferences;
    FixedVector<StructureID> m_weakStructureReferences;
    FixedVector<CatchEntrypointData> m_catchEntrypoints;
    FixedVector<CodeBlockJettisoningWatchpoint> m_watchpoints;
    FixedVector<AdaptiveStructureWatchpoint> m_adaptiveStructureWatchpoints;
    FixedVector<AdaptiveInferredPropertyValueWatchpoint> m_adaptiveInferredPropertyValueWatchpoints;
    std::unique_ptr<PCToCodeOriginMap> m_pcToCodeOriginMap;
    std::unique_ptr<RecordedStatuses> recordedStatuses;
    FixedVector<JumpReplacement> m_jumpReplacements;
    FixedVector<std::unique_ptr<BoyerMooreHorspoolTable<uint8_t>>> m_stringSearchTable8;
    Bag<StructureStubInfo> m_stubInfos;
    Bag<OptimizingCallLinkInfo> m_callLinkInfos;
    Bag<DirectCallLinkInfo> m_directCallLinkInfos;
    Yarr::YarrBoyerMooreData m_boyerMooreData;
    
    ScratchBuffer* catchOSREntryBuffer;
    RefPtr<Profiler::Compilation> compilation;
    
#if USE(JSVALUE32_64)
    Bag<double> doubleConstants;
#endif
    
    unsigned frameRegisterCount { std::numeric_limits<unsigned>::max() };
    unsigned requiredRegisterCountForExit { std::numeric_limits<unsigned>::max() };

private:
    bool m_isUnlinked { false };
    bool m_isStillValid { true };
    bool m_hasVMTrapsBreakpointsInstalled { false };
};

CodeBlock* codeBlockForVMTrapPC(void* pc);

} } // namespace JSC::DFG

#endif // ENABLE(DFG_JIT)
