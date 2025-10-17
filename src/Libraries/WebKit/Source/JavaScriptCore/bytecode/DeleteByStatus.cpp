/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 6, 2025.
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
#include "DeleteByStatus.h"

#include "CacheableIdentifierInlines.h"
#include "CodeBlock.h"
#include "ICStatusUtils.h"
#include "InlineCacheCompiler.h"
#include "StructureStubInfo.h"
#include <wtf/ListDump.h>
#include <wtf/TZoneMallocInlines.h>

namespace JSC {

WTF_MAKE_TZONE_ALLOCATED_IMPL(DeleteByStatus);

bool DeleteByStatus::appendVariant(const DeleteByVariant& variant)
{
    return appendICStatusVariant(m_variants, variant);
}

void DeleteByStatus::shrinkToFit()
{
    m_variants.shrinkToFit();
}

DeleteByStatus DeleteByStatus::computeForBaseline(CodeBlock* baselineBlock, ICStatusMap& map, BytecodeIndex bytecodeIndex, ExitFlag didExit)
{
    ConcurrentJSLocker locker(baselineBlock->m_lock);

    DeleteByStatus result;

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
DeleteByStatus::DeleteByStatus(StubInfoSummary summary, StructureStubInfo& stubInfo)
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

DeleteByStatus DeleteByStatus::computeForStubInfoWithoutExitSiteFeedback(const ConcurrentJSLocker& locker, CodeBlock* block, StructureStubInfo* stubInfo)
{
    StubInfoSummary summary = StructureStubInfo::summary(locker, block->vm(), stubInfo);
    if (!isInlineable(summary))
        return DeleteByStatus(summary, *stubInfo);

    DeleteByStatus result;
    result.m_state = Simple;
    switch (stubInfo->cacheType()) {
    case CacheType::Unset:
        return DeleteByStatus(NoInformation);

    case CacheType::Stub: {
        auto list = stubInfo->listedAccessCases(locker);
        for (unsigned listIndex = 0; listIndex < list.size(); ++listIndex) {
            const AccessCase& access = *list.at(listIndex);
            ASSERT(!access.viaGlobalProxy());

            Structure* structure = access.structure();
            ASSERT(structure);

            switch (access.type()) {
            case AccessCase::DeleteMiss:
            case AccessCase::DeleteNonConfigurable: {
                DeleteByVariant variant(access.identifier(), access.type() == AccessCase::DeleteMiss ? true : false, structure, nullptr, invalidOffset);
                if (!result.appendVariant(variant))
                    return DeleteByStatus(JSC::slowVersion(summary), *stubInfo);
                break;
            }
            case AccessCase::Delete: {
                PropertyOffset offset;
                Structure* newStructure = Structure::removePropertyTransitionFromExistingStructureConcurrently(structure, access.identifier().uid(), offset);
                if (!newStructure)
                    return DeleteByStatus(JSC::slowVersion(summary), *stubInfo);
                ASSERT_UNUSED(offset, offset == access.offset());                
                DeleteByVariant variant(access.identifier(), true, structure, newStructure, access.offset());

                if (!result.appendVariant(variant))
                    return DeleteByStatus(JSC::slowVersion(summary), *stubInfo);
                break;
            }
            default:
                ASSERT_NOT_REACHED();
                return DeleteByStatus(JSC::slowVersion(summary), *stubInfo);
            }
        }

        result.shrinkToFit();
        return result;
    }

    default:
        return DeleteByStatus(JSC::slowVersion(summary), *stubInfo);
    }

    RELEASE_ASSERT_NOT_REACHED();
    return DeleteByStatus();
}

DeleteByStatus DeleteByStatus::computeFor(
    CodeBlock* baselineBlock, ICStatusMap& baselineMap,
    ICStatusContextStack& contextStack, CodeOrigin codeOrigin)
{
    BytecodeIndex bytecodeIndex = codeOrigin.bytecodeIndex();
    ExitFlag didExit = hasBadCacheExitSite(baselineBlock, bytecodeIndex);

    for (ICStatusContext* context : contextStack) {
        ICStatus status = context->get(codeOrigin);

        auto bless = [&] (const DeleteByStatus& result) -> DeleteByStatus {
            if (!context->isInlined(codeOrigin)) {
                DeleteByStatus baselineResult = computeForBaseline(
                    baselineBlock, baselineMap, bytecodeIndex, didExit);
                baselineResult.merge(result);
                return baselineResult;
            }
            if (didExit.isSet(ExitFromInlined))
                return result.slowVersion();
            return result;
        };

        if (status.stubInfo) {
            DeleteByStatus result;
            {
                ConcurrentJSLocker locker(context->optimizedCodeBlock->m_lock);
                result = computeForStubInfoWithoutExitSiteFeedback(
                    locker, context->optimizedCodeBlock, status.stubInfo);
            }
            if (result.isSet())
                return bless(result);
        }

        if (status.deleteStatus)
            return bless(*status.deleteStatus);
    }

    return computeForBaseline(baselineBlock, baselineMap, bytecodeIndex, didExit);
}

#endif // ENABLE(JIT)

DeleteByStatus DeleteByStatus::slowVersion() const
{
    if (observedSlowPath())
        return DeleteByStatus(ObservedTakesSlowPath);
    return DeleteByStatus(LikelyTakesSlowPath);
}

void DeleteByStatus::merge(const DeleteByStatus& other)
{
    if (other.m_state == NoInformation)
        return;

    auto mergeSlow = [&] () {
        if (observedSlowPath() || other.observedSlowPath())
            *this = DeleteByStatus(ObservedTakesSlowPath);
        else
            *this = DeleteByStatus(LikelyTakesSlowPath);
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

void DeleteByStatus::filter(const StructureSet& set)
{
    if (m_state != Simple)
        return;
    m_variants.removeAllMatching(
        [&] (auto& variant) -> bool {
            return !set.contains(variant.oldStructure());
        });
    if (m_variants.isEmpty())
        m_state = NoInformation;
}

CacheableIdentifier DeleteByStatus::singleIdentifier() const
{
    return singleIdentifierForICStatus(m_variants);
}

template<typename Visitor>
void DeleteByStatus::visitAggregateImpl(Visitor& visitor)
{
    for (DeleteByVariant& variant : m_variants)
        variant.visitAggregate(visitor);
}

DEFINE_VISIT_AGGREGATE(DeleteByStatus);

template<typename Visitor>
void DeleteByStatus::markIfCheap(Visitor& visitor)
{
    for (DeleteByVariant& variant : m_variants)
        variant.markIfCheap(visitor);
}

template void DeleteByStatus::markIfCheap(AbstractSlotVisitor&);
template void DeleteByStatus::markIfCheap(SlotVisitor&);

bool DeleteByStatus::finalize(VM& vm)
{
    for (auto& variant : m_variants) {
        if (!variant.finalize(vm))
            return false;
    }
    return true;
}

void DeleteByStatus::dump(PrintStream& out) const
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
