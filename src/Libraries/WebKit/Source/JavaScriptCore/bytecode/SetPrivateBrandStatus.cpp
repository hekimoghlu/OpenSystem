/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 5, 2025.
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
#include "SetPrivateBrandStatus.h"

#include "CacheableIdentifierInlines.h"
#include "CodeBlock.h"
#include "ICStatusUtils.h"
#include "InlineCacheCompiler.h"
#include "StructureStubInfo.h"
#include <wtf/ListDump.h>
#include <wtf/TZoneMallocInlines.h>

namespace JSC {

WTF_MAKE_TZONE_ALLOCATED_IMPL(SetPrivateBrandStatus);

bool SetPrivateBrandStatus::appendVariant(const SetPrivateBrandVariant& variant)
{
    return appendICStatusVariant(m_variants, variant);
}

void SetPrivateBrandStatus::shrinkToFit()
{
    m_variants.shrinkToFit();
}

SetPrivateBrandStatus SetPrivateBrandStatus::computeForBaseline(CodeBlock* baselineBlock, ICStatusMap& map, BytecodeIndex bytecodeIndex, ExitFlag didExit)
{
    ConcurrentJSLocker locker(baselineBlock->m_lock);

    SetPrivateBrandStatus result;

#if ENABLE(DFG_JIT)
    result = computeForStubInfoWithoutExitSiteFeedback(
        locker, baselineBlock, map.get(CodeOrigin(bytecodeIndex)).stubInfo);

    if (didExit)
        return result.slowVersion();
#else
    UNUSED_PARAM(map);
    UNUSED_PARAM(didExit);
    UNUSED_PARAM(bytecodeIndex);
#endif

    return result;
}

#if ENABLE(JIT)
SetPrivateBrandStatus::SetPrivateBrandStatus(StubInfoSummary summary, StructureStubInfo& stubInfo)
{
    switch (summary) {
    case StubInfoSummary::NoInformation:
        m_state = NoInformation;
        return;
    case StubInfoSummary::Simple:
    case StubInfoSummary::Megamorphic:
    case StubInfoSummary::MakesCalls:
    case StubInfoSummary::TakesSlowPathAndMakesCalls:
        RELEASE_ASSERT_NOT_REACHED();
        return;
    case StubInfoSummary::TakesSlowPath:
        m_state = stubInfo.tookSlowPath ? ObservedTakesSlowPath : LikelyTakesSlowPath;
        return;
    }
    RELEASE_ASSERT_NOT_REACHED();
}

SetPrivateBrandStatus SetPrivateBrandStatus::computeForStubInfoWithoutExitSiteFeedback(const ConcurrentJSLocker& locker, CodeBlock* block, StructureStubInfo* stubInfo)
{
    StubInfoSummary summary = StructureStubInfo::summary(locker, block->vm(), stubInfo);
    if (!isInlineable(summary))
        return SetPrivateBrandStatus(summary, *stubInfo);

    SetPrivateBrandStatus result;
    result.m_state = Simple;
    switch (stubInfo->cacheType()) {
    case CacheType::Unset:
        return SetPrivateBrandStatus(NoInformation);

    case CacheType::Stub: {
        auto list = stubInfo->listedAccessCases(locker);

        for (unsigned listIndex = 0; listIndex < list.size(); ++listIndex) {
            const AccessCase& access = *list.at(listIndex);

            Structure* structure = access.structure();
            ASSERT(structure);
            ASSERT(access.type() == AccessCase::SetPrivateBrand);

            Structure* newStructure = Structure::setBrandTransitionFromExistingStructureConcurrently(structure, access.identifier().uid());
            if (!newStructure)
                return SetPrivateBrandStatus(JSC::slowVersion(summary), *stubInfo);

            SetPrivateBrandVariant variant(access.identifier(), structure, newStructure);
            if (!result.appendVariant(variant))
                return SetPrivateBrandStatus(JSC::slowVersion(summary), *stubInfo);
        }

        result.shrinkToFit();
        return result;
    }

    default:
        return SetPrivateBrandStatus(JSC::slowVersion(summary), *stubInfo);
    }

    RELEASE_ASSERT_NOT_REACHED();
    return SetPrivateBrandStatus();
}

SetPrivateBrandStatus SetPrivateBrandStatus::computeFor(
    CodeBlock* baselineBlock, ICStatusMap& baselineMap,
    ICStatusContextStack& contextStack, CodeOrigin codeOrigin)
{
    BytecodeIndex bytecodeIndex = codeOrigin.bytecodeIndex();
    ExitFlag didExit = hasBadCacheExitSite(baselineBlock, bytecodeIndex);

    for (ICStatusContext* context : contextStack) {
        ICStatus status = context->get(codeOrigin);

        auto bless = [&] (const SetPrivateBrandStatus& result) -> SetPrivateBrandStatus {
            if (!context->isInlined(codeOrigin)) {
                SetPrivateBrandStatus baselineResult = computeForBaseline(
                    baselineBlock, baselineMap, bytecodeIndex, didExit);
                baselineResult.merge(result);
                return baselineResult;
            }
            if (didExit.isSet(ExitFromInlined))
                return result.slowVersion();
            return result;
        };

        if (status.stubInfo) {
            SetPrivateBrandStatus result;
            {
                ConcurrentJSLocker locker(context->optimizedCodeBlock->m_lock);
                result = computeForStubInfoWithoutExitSiteFeedback(
                    locker, context->optimizedCodeBlock, status.stubInfo);
            }
            if (result.isSet())
                return bless(result);
        }
    }

    return computeForBaseline(baselineBlock, baselineMap, bytecodeIndex, didExit);
}

#endif // ENABLE(JIT)

SetPrivateBrandStatus SetPrivateBrandStatus::slowVersion() const
{
    if (observedSlowPath())
        return SetPrivateBrandStatus(ObservedTakesSlowPath);
    return SetPrivateBrandStatus(LikelyTakesSlowPath);
}

void SetPrivateBrandStatus::merge(const SetPrivateBrandStatus& other)
{
    if (other.m_state == NoInformation)
        return;

    auto mergeSlow = [&] () {
        if (observedSlowPath() || other.observedSlowPath())
            *this = SetPrivateBrandStatus(ObservedTakesSlowPath);
        else
            *this = SetPrivateBrandStatus(LikelyTakesSlowPath);
    };

    switch (m_state) {
    case NoInformation:
        *this = other;
        return;

    case Simple:
        if (m_state != other.m_state)
            return mergeSlow();

        for (auto& otherVariant : other.m_variants) {
            if (!appendVariant(otherVariant))
                return mergeSlow();
        }
        shrinkToFit();
        return;

    case LikelyTakesSlowPath:
    case ObservedTakesSlowPath:
        return mergeSlow();
    }

    RELEASE_ASSERT_NOT_REACHED();
}

void SetPrivateBrandStatus::filter(const StructureSet& structureSet)
{
    if (m_state != Simple)
        return;
    m_variants.removeAllMatching(
        [&] (auto& variant) -> bool {
            return !structureSet.contains(variant.oldStructure());
        });
    if (m_variants.isEmpty())
        m_state = NoInformation;
}

CacheableIdentifier SetPrivateBrandStatus::singleIdentifier() const
{
    return singleIdentifierForICStatus(m_variants);
}

template<typename Visitor>
void SetPrivateBrandStatus::visitAggregateImpl(Visitor& visitor)
{
    for (SetPrivateBrandVariant& variant : m_variants)
        variant.visitAggregate(visitor);
}

DEFINE_VISIT_AGGREGATE(SetPrivateBrandStatus);

template<typename Visitor>
void SetPrivateBrandStatus::markIfCheap(Visitor& visitor)
{
    for (SetPrivateBrandVariant& variant : m_variants)
        variant.markIfCheap(visitor);
}

template void SetPrivateBrandStatus::markIfCheap(AbstractSlotVisitor&);
template void SetPrivateBrandStatus::markIfCheap(SlotVisitor&);

bool SetPrivateBrandStatus::finalize(VM& vm)
{
    for (auto& variant : m_variants) {
        if (!variant.finalize(vm))
            return false;
    }
    return true;
}

void SetPrivateBrandStatus::dump(PrintStream& out) const
{
    out.print("(");
    switch (m_state) {
    case NoInformation:
        out.print("NoInformation");
        break;
    case Simple:
        out.print("Simple");
        break;
    case LikelyTakesSlowPath:
        out.print("LikelyTakesSlowPath");
        break;
    case ObservedTakesSlowPath:
        out.print("ObservedTakesSlowPath");
        break;
    }
    out.print(", ", listDump(m_variants), ")");
}

} // namespace JSC
