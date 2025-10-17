/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 20, 2023.
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
#include "DFGCommonData.h"

#if ENABLE(DFG_JIT)

#include "CodeBlock.h"
#include "DFGNode.h"
#include "DFGPlan.h"
#include "InlineCallFrame.h"
#include "JSCJSValueInlines.h"
#include "TrackedReferences.h"
#include <wtf/Lock.h>
#include <wtf/NeverDestroyed.h>

namespace JSC { namespace DFG {

void CommonData::shrinkToFit()
{
    codeOrigins->shrinkToFit();
}

static Lock pcCodeBlockMapLock;
inline UncheckedKeyHashMap<void*, CodeBlock*>& pcCodeBlockMap() WTF_REQUIRES_LOCK(pcCodeBlockMapLock)
{
    static LazyNeverDestroyed<UncheckedKeyHashMap<void*, CodeBlock*>> pcCodeBlockMap;
    static std::once_flag onceKey;
    std::call_once(onceKey, [&] {
        pcCodeBlockMap.construct();
    });
    return pcCodeBlockMap;
}

bool CommonData::invalidateLinkedCode()
{
    if (m_isUnlinked) {
        ASSERT(m_jumpReplacements.isEmpty());
        return true;
    }

    if (!m_isStillValid)
        return false;

    if (UNLIKELY(m_hasVMTrapsBreakpointsInstalled)) {
        Locker locker { pcCodeBlockMapLock };
        auto& map = pcCodeBlockMap();
        for (auto& jumpReplacement : m_jumpReplacements)
            map.remove(jumpReplacement.dataLocation());
        m_hasVMTrapsBreakpointsInstalled = false;
    }

    for (unsigned i = m_jumpReplacements.size(); i--;)
        m_jumpReplacements[i].fire();

    m_isStillValid = false;
    return true;
}

CommonData::~CommonData()
{
    if (m_isUnlinked)
        return;
    if (UNLIKELY(m_hasVMTrapsBreakpointsInstalled)) {
        Locker locker { pcCodeBlockMapLock };
        auto& map = pcCodeBlockMap();
        for (auto& jumpReplacement : m_jumpReplacements)
            map.remove(jumpReplacement.dataLocation());
    }
}

void CommonData::installVMTrapBreakpoints(CodeBlock* owner)
{
    ASSERT(!m_isUnlinked);
    Locker locker { pcCodeBlockMapLock };
    if (!m_isStillValid || m_hasVMTrapsBreakpointsInstalled)
        return;
    m_hasVMTrapsBreakpointsInstalled = true;

    auto& map = pcCodeBlockMap();
#if !defined(NDEBUG)
    // We need to be able to handle more than one invalidation point at the same pc
    // but we want to make sure we don't forget to remove a pc from the map.
    UncheckedKeyHashSet<void*> newReplacements;
#endif
    for (auto& jumpReplacement : m_jumpReplacements) {
        jumpReplacement.installVMTrapBreakpoint();
        void* source = jumpReplacement.dataLocation();
        auto result = map.add(source, owner);
        UNUSED_PARAM(result);
#if !defined(NDEBUG)
        ASSERT(result.isNewEntry || newReplacements.contains(source));
        newReplacements.add(source);
#endif
    }
}

CodeBlock* codeBlockForVMTrapPC(void* pc)
{
    ASSERT(isJITPC(pc));
    Locker locker { pcCodeBlockMapLock };
    auto& map = pcCodeBlockMap();
    auto result = map.find(pc);
    if (result == map.end())
        return nullptr;
    return result->value;
}

void CommonData::validateReferences(const TrackedReferences& trackedReferences)
{
    if (InlineCallFrameSet* set = inlineCallFrames.get()) {
        for (InlineCallFrame* inlineCallFrame : *set) {
            for (ValueRecovery& recovery : inlineCallFrame->m_argumentsWithFixup) {
                if (recovery.isConstant())
                    trackedReferences.check(recovery.constant());
            }
            
            if (CodeBlock* baselineCodeBlock = inlineCallFrame->baselineCodeBlock.get())
                trackedReferences.check(baselineCodeBlock);
            
            if (inlineCallFrame->calleeRecovery.isConstant())
                trackedReferences.check(inlineCallFrame->calleeRecovery.constant());
        }
    }
    
    for (auto& watchpoint : m_adaptiveStructureWatchpoints)
        watchpoint.key().validateReferences(trackedReferences);
}

void CommonData::finalizeCatchEntrypoints(Vector<CatchEntrypointData>&& catchEntrypoints)
{
    std::sort(catchEntrypoints.begin(), catchEntrypoints.end(),
        [] (const CatchEntrypointData& a, const CatchEntrypointData& b) { return a.bytecodeIndex < b.bytecodeIndex; });
    ASSERT(m_catchEntrypoints.isEmpty());
    m_catchEntrypoints = WTFMove(catchEntrypoints);

#if ASSERT_ENABLED
    for (unsigned i = 0; i + 1 < m_catchEntrypoints.size(); ++i)
        ASSERT(m_catchEntrypoints[i].bytecodeIndex <= m_catchEntrypoints[i + 1].bytecodeIndex);
#endif
}

void CommonData::clearWatchpoints()
{
    m_watchpoints = FixedVector<CodeBlockJettisoningWatchpoint>();
    m_adaptiveStructureWatchpoints = FixedVector<AdaptiveStructureWatchpoint>();
    m_adaptiveInferredPropertyValueWatchpoints = FixedVector<AdaptiveInferredPropertyValueWatchpoint>();
}

} } // namespace JSC::DFG

#endif // ENABLE(DFG_JIT)

