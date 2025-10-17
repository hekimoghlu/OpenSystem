/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 18, 2022.
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
#include "SetPrivateBrandVariant.h"

#include "CacheableIdentifierInlines.h"

namespace JSC {

SetPrivateBrandVariant::SetPrivateBrandVariant(CacheableIdentifier identifier, Structure* oldStructure, Structure* newStructure)
    : m_oldStructure(oldStructure)
    , m_newStructure(newStructure)
    , m_identifier(WTFMove(identifier))
{ }

SetPrivateBrandVariant::~SetPrivateBrandVariant() = default;

bool SetPrivateBrandVariant::attemptToMerge(const SetPrivateBrandVariant& other)
{
    if (!!m_identifier != !!other.m_identifier)
        return false;

    if (m_identifier && (m_identifier != other.m_identifier))
        return false;

    if (m_oldStructure != other.m_oldStructure)
        return false;
    ASSERT(m_newStructure == other.m_newStructure);

    return true;
}

template<typename Visitor>
void SetPrivateBrandVariant::markIfCheap(Visitor& visitor)
{
    if (m_oldStructure)
        m_oldStructure->markIfCheap(visitor);
    if (m_newStructure)
        m_newStructure->markIfCheap(visitor);
}

template void SetPrivateBrandVariant::markIfCheap(AbstractSlotVisitor&);
template void SetPrivateBrandVariant::markIfCheap(SlotVisitor&);

bool SetPrivateBrandVariant::finalize(VM& vm)
{
    if (!vm.heap.isMarked(m_oldStructure))
        return false;
    if (m_newStructure && !vm.heap.isMarked(m_newStructure))
        return false;
    return true;
}

template<typename Visitor>
void SetPrivateBrandVariant::visitAggregateImpl(Visitor& visitor)
{
    m_identifier.visitAggregate(visitor);
}

DEFINE_VISIT_AGGREGATE(SetPrivateBrandVariant);

void SetPrivateBrandVariant::dump(PrintStream& out) const
{
    dumpInContext(out, nullptr);
}

void SetPrivateBrandVariant::dumpInContext(PrintStream& out, DumpContext*) const
{
    out.print("<");
    out.print("id='", m_identifier, "'");
    if (m_oldStructure)
        out.print(", ", *m_oldStructure);
    if (m_newStructure)
        out.print(" -> ", *m_newStructure);
    out.print(">");
}

} // namespace JSC
