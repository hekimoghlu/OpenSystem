/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 25, 2023.
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
#include "DFGAdaptiveStructureWatchpoint.h"

#if ENABLE(DFG_JIT)

#include "CodeBlockInlines.h"
#include "JSCellInlines.h"

namespace JSC { namespace DFG {

AdaptiveStructureWatchpoint::AdaptiveStructureWatchpoint(const ObjectPropertyCondition& key, CodeBlock* codeBlock)
    : Watchpoint(Watchpoint::Type::AdaptiveStructure)
    , m_codeBlock(codeBlock)
    , m_key(key)
{
    RELEASE_ASSERT(key.watchingRequiresStructureTransitionWatchpoint());
    RELEASE_ASSERT(!key.watchingRequiresReplacementWatchpoint());
}

AdaptiveStructureWatchpoint::AdaptiveStructureWatchpoint()
    : Watchpoint(Watchpoint::Type::AdaptiveStructure)
    , m_codeBlock(nullptr)
{
}

void AdaptiveStructureWatchpoint::initialize(const ObjectPropertyCondition& key, CodeBlock* codeBlock)
{
    m_codeBlock = codeBlock;
    m_key = key;
    RELEASE_ASSERT(key.watchingRequiresStructureTransitionWatchpoint());
    RELEASE_ASSERT(!key.watchingRequiresReplacementWatchpoint());
}

void AdaptiveStructureWatchpoint::install(VM&)
{
    RELEASE_ASSERT(m_key.isWatchable(PropertyCondition::MakeNoChanges));
    
    m_key.object()->structure()->addTransitionWatchpoint(this);
}

void AdaptiveStructureWatchpoint::fireInternal(VM& vm, const FireDetail& detail)
{
    ASSERT(!m_codeBlock->wasDestructed());
    if (m_codeBlock->isPendingDestruction())
        return;

    if (m_key.isWatchable(PropertyCondition::EnsureWatchability)) {
        install(vm);
        return;
    }
    
    if (DFG::shouldDumpDisassembly()) {
        dataLog(
            "Firing watchpoint ", RawPointer(this), " (", m_key, ") on ", *m_codeBlock, "\n");
    }

    auto lambda = scopedLambda<void(PrintStream&)>([&](PrintStream& out) {
        out.print("Adaptation of ", m_key, " failed: ", detail);
    });
    LazyFireDetail lazyDetail(lambda);
    m_codeBlock->jettison(Profiler::JettisonDueToUnprofiledWatchpoint, CountReoptimization, &lazyDetail);
}

} } // namespace JSC::DFG

#endif // ENABLE(DFG_JIT)

