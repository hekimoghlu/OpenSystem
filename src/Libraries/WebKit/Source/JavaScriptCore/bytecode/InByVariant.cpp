/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 12, 2025.
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
#include "InByVariant.h"

#include "CacheableIdentifierInlines.h"

namespace JSC {

InByVariant::InByVariant(CacheableIdentifier identifier, const StructureSet& structureSet, PropertyOffset offset, const ObjectPropertyConditionSet& conditionSet, std::unique_ptr<CallLinkStatus> callLinkStatus)
    : m_structureSet(structureSet)
    , m_conditionSet(conditionSet)
    , m_offset(offset)
    , m_identifier(WTFMove(identifier))
    , m_callLinkStatus(WTFMove(callLinkStatus))
{
    if (!structureSet.size()) {
        ASSERT(offset == invalidOffset);
        ASSERT(conditionSet.isEmpty());
    }
}

InByVariant::~InByVariant() = default;

InByVariant::InByVariant(const InByVariant& other)
    : InByVariant(other.m_identifier)
{
    *this = other;
}

InByVariant& InByVariant::operator=(const InByVariant& other)
{
    m_identifier = other.m_identifier;
    m_structureSet = other.m_structureSet;
    m_conditionSet = other.m_conditionSet;
    m_offset = other.m_offset;
    if (other.m_callLinkStatus)
        m_callLinkStatus = makeUnique<CallLinkStatus>(*other.m_callLinkStatus);
    else
        m_callLinkStatus = nullptr;
    return *this;
}

bool InByVariant::attemptToMerge(const InByVariant& other)
{
    if (!!m_identifier != !!other.m_identifier)
        return false;

    if (m_identifier && (m_identifier != other.m_identifier))
        return false;

    if (m_offset != other.m_offset)
        return false;

    if (m_callLinkStatus || other.m_callLinkStatus) {
        if (!(m_callLinkStatus && other.m_callLinkStatus))
            return false;
    }

    if (m_conditionSet.isEmpty() != other.m_conditionSet.isEmpty())
        return false;

    ObjectPropertyConditionSet mergedConditionSet;
    if (!m_conditionSet.isEmpty()) {
        mergedConditionSet = m_conditionSet.mergedWith(other.m_conditionSet);
        if (!mergedConditionSet.isValid())
            return false;
        // If this is a hit variant, one slot base should exist. If this is not a hit variant, the slot base is not necessary.
        if (isHit() && !mergedConditionSet.hasOneSlotBaseCondition())
            return false;
    }
    m_conditionSet = mergedConditionSet;

    m_structureSet.merge(other.m_structureSet);

    if (m_callLinkStatus)
        m_callLinkStatus->merge(*other.m_callLinkStatus);

    return true;
}

template<typename Visitor>
void InByVariant::visitAggregateImpl(Visitor& visitor)
{
    m_identifier.visitAggregate(visitor);
}

DEFINE_VISIT_AGGREGATE(InByVariant);

template<typename Visitor>
void InByVariant::markIfCheap(Visitor& visitor)
{
    m_structureSet.markIfCheap(visitor);
}

template void InByVariant::markIfCheap(AbstractSlotVisitor&);
template void InByVariant::markIfCheap(SlotVisitor&);

bool InByVariant::finalize(VM& vm)
{
    if (!m_structureSet.isStillAlive(vm))
        return false;
    if (!m_conditionSet.areStillLive(vm))
        return false;
    if (m_callLinkStatus && !m_callLinkStatus->finalize(vm))
        return false;
    return true;
}

void InByVariant::dump(PrintStream& out) const
{
    dumpInContext(out, nullptr);
}

void InByVariant::dumpInContext(PrintStream& out, DumpContext* context) const
{
    out.print("<id='", m_identifier, "', ");
    if (!isSet()) {
        out.print("empty>");
        return;
    }

    out.print(inContext(structureSet(), context), ", ", inContext(m_conditionSet, context));
    out.print(", offset = ", offset());
    if (m_callLinkStatus)
        out.print(", call = ", *m_callLinkStatus);
    out.print(">");
}

} // namespace JSC

