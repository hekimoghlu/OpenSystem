/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 20, 2022.
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
#include "InstanceOfStatus.h"

#include "ICStatusUtils.h"
#include "InlineCacheCompiler.h"
#include "InstanceOfAccessCase.h"
#include "JSCellInlines.h"
#include "StructureStubInfo.h"

namespace JSC {

void InstanceOfStatus::appendVariant(const InstanceOfVariant& variant)
{
    appendICStatusVariant(m_variants, variant);
}

void InstanceOfStatus::shrinkToFit()
{
    m_variants.shrinkToFit();
}

InstanceOfStatus InstanceOfStatus::computeFor(
    CodeBlock* codeBlock, ICStatusMap& infoMap, BytecodeIndex bytecodeIndex)
{
    ConcurrentJSLocker locker(codeBlock->m_lock);
    
    InstanceOfStatus result;
#if ENABLE(DFG_JIT)
    result = computeForStubInfo(locker, codeBlock->vm(), infoMap.get(CodeOrigin(bytecodeIndex)).stubInfo);

    if (!result.takesSlowPath()) {
        UnlinkedCodeBlock* unlinkedCodeBlock = codeBlock->unlinkedCodeBlock();
        ConcurrentJSLocker locker(unlinkedCodeBlock->m_lock);
        // We also check for BadType here in case this is "primitive instanceof Foo".
        if (unlinkedCodeBlock->hasExitSite(locker, DFG::FrequentExitSite(bytecodeIndex, BadCache))
            || unlinkedCodeBlock->hasExitSite(locker, DFG::FrequentExitSite(bytecodeIndex, BadConstantCache))
            || unlinkedCodeBlock->hasExitSite(locker, DFG::FrequentExitSite(bytecodeIndex, BadType)))
            return TakesSlowPath;
    }
#else
    UNUSED_PARAM(infoMap);
    UNUSED_PARAM(bytecodeIndex);
#endif
    
    return result;
}

#if ENABLE(DFG_JIT)
InstanceOfStatus InstanceOfStatus::computeForStubInfo(const ConcurrentJSLocker& locker, VM& vm, StructureStubInfo* stubInfo)
{
    // FIXME: We wouldn't have to bail for nonCell if we taught MatchStructure how to handle non
    // cells. If we fixed that then we wouldn't be able to use summary();
    // https://bugs.webkit.org/show_bug.cgi?id=185784
    StubInfoSummary summary = StructureStubInfo::summary(locker, vm, stubInfo);
    if (!isInlineable(summary))
        return InstanceOfStatus(summary);
    
    if (stubInfo->cacheType() != CacheType::Stub)
        return TakesSlowPath; // This is conservative. It could be that we have no information.
    
    InstanceOfStatus result;
    auto list = stubInfo->listedAccessCases(locker);
    for (unsigned listIndex = 0; listIndex < list.size(); ++listIndex) {
        const AccessCase& access = *list.at(listIndex);
        
        if (access.type() == AccessCase::InstanceOfMegamorphic)
            return Megamorphic;
        
        if (!access.conditionSet().structuresEnsureValidity())
            return TakesSlowPath;
        
        result.appendVariant(InstanceOfVariant(
            access.structure(),
            access.conditionSet(),
            access.as<InstanceOfAccessCase>().prototype(),
            access.type() == AccessCase::InstanceOfHit));
    }
    
    result.shrinkToFit();
    return result;
}
#endif // ENABLE(DFG_JIT)

JSObject* InstanceOfStatus::commonPrototype() const
{
    JSObject* prototype = nullptr;
    for (const InstanceOfVariant& variant : m_variants) {
        if (!prototype) {
            prototype = variant.prototype();
            continue;
        }
        if (prototype != variant.prototype())
            return nullptr;
    }
    return prototype;
}

void InstanceOfStatus::filter(const StructureSet& structureSet)
{
    if (m_state != Simple)
        return;
    filterICStatusVariants(m_variants, structureSet);
    if (m_variants.isEmpty())
        m_state = NoInformation;
}

} // namespace JSC

