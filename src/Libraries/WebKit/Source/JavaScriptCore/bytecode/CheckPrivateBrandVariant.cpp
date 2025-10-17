/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 18, 2023.
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
#include "CheckPrivateBrandVariant.h"

#include "CacheableIdentifierInlines.h"

namespace JSC {

CheckPrivateBrandVariant::CheckPrivateBrandVariant(CacheableIdentifier identifier, const StructureSet& structureSet)
    : m_structureSet(structureSet)
    , m_identifier(WTFMove(identifier))
{ }

CheckPrivateBrandVariant::~CheckPrivateBrandVariant() = default;

bool CheckPrivateBrandVariant::attemptToMerge(const CheckPrivateBrandVariant& other)
{
    if (!!m_identifier != !!other.m_identifier)
        return false;

    if (m_identifier && (m_identifier != other.m_identifier))
        return false;

    m_structureSet.merge(other.m_structureSet);

    return true;
}

template<typename Visitor>
void CheckPrivateBrandVariant::markIfCheap(Visitor& visitor)
{
    m_structureSet.markIfCheap(visitor);
}

template void CheckPrivateBrandVariant::markIfCheap(AbstractSlotVisitor&);
template void CheckPrivateBrandVariant::markIfCheap(SlotVisitor&);

bool CheckPrivateBrandVariant::finalize(VM& vm)
{
    if (!m_structureSet.isStillAlive(vm))
        return false;
    return true;
}

template<typename Visitor>
void CheckPrivateBrandVariant::visitAggregateImpl(Visitor& visitor)
{
    m_identifier.visitAggregate(visitor);
}

DEFINE_VISIT_AGGREGATE(CheckPrivateBrandVariant);

void CheckPrivateBrandVariant::dump(PrintStream& out) const
{
    dumpInContext(out, nullptr);
}

void CheckPrivateBrandVariant::dumpInContext(PrintStream& out, DumpContext* context) const
{
    out.print("<id='", m_identifier, "', ", inContext(structureSet(), context), ">");
}

} // namespace JSC
