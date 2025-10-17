/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 11, 2025.
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
#include "DFGAdaptiveInferredPropertyValueWatchpoint.h"

#if ENABLE(DFG_JIT)

#include "CodeBlock.h"
#include "DFGCommon.h"
#include <wtf/TZoneMallocInlines.h>

namespace JSC { namespace DFG {

WTF_MAKE_TZONE_ALLOCATED_IMPL(AdaptiveInferredPropertyValueWatchpoint);

AdaptiveInferredPropertyValueWatchpoint::AdaptiveInferredPropertyValueWatchpoint(const ObjectPropertyCondition& key, CodeBlock* codeBlock)
    : Base(key)
    , m_codeBlock(codeBlock)
{
}

void AdaptiveInferredPropertyValueWatchpoint::initialize(const ObjectPropertyCondition& key, CodeBlock* codeBlock)
{
    Base::initialize(key);
    m_codeBlock = codeBlock;
}

void AdaptiveInferredPropertyValueWatchpoint::handleFire(VM&, const FireDetail& detail)
{
    if (DFG::shouldDumpDisassembly())
        dataLog("Firing watchpoint ", RawPointer(this), " (", key(), ") on ", *m_codeBlock, "\n");


    auto lambda = scopedLambda<void(PrintStream&)>([&](PrintStream& out) {
        out.print("Adaptation of ", key(), " failed: ", detail);
    });
    LazyFireDetail lazyDetail(lambda);
    m_codeBlock->jettison(Profiler::JettisonDueToUnprofiledWatchpoint, CountReoptimization, &lazyDetail);
}

bool AdaptiveInferredPropertyValueWatchpoint::isValid() const
{
    ASSERT(!m_codeBlock->wasDestructed());
    return !m_codeBlock->isPendingDestruction();
}

} } // namespace JSC::DFG

#endif // ENABLE(DFG_JIT)
