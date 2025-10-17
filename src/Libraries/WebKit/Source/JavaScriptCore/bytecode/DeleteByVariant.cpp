/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 5, 2022.
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
#include "DeleteByVariant.h"

#include "CacheableIdentifierInlines.h"

namespace JSC {

DeleteByVariant::DeleteByVariant(CacheableIdentifier identifier, bool result, Structure* oldStructure, Structure* newStructure, PropertyOffset offset)
    : m_result(result)
    , m_oldStructure(oldStructure)
    , m_newStructure(newStructure)
    , m_offset(offset)
    , m_identifier(WTFMove(identifier))
{
    ASSERT(oldStructure);
    if (m_offset == invalidOffset)
        ASSERT(!newStructure);
    else
        ASSERT(newStructure);
}

DeleteByVariant::~DeleteByVariant() = default;

DeleteByVariant::DeleteByVariant(const DeleteByVariant& other)
{
    *this = other;
}

DeleteByVariant& DeleteByVariant::operator=(const DeleteByVariant& other)
{
    m_identifier = other.m_identifier;
    m_result = other.m_result;
    m_oldStructure = other.m_oldStructure;
    m_newStructure = other.m_newStructure;
    m_offset = other.m_offset;
    return *this;
}

bool DeleteByVariant::attemptToMerge(const DeleteByVariant& other)
{
    if (!!m_identifier != !!other.m_identifier)
        return false;

    if (m_result != other.m_result)
        return false;

    if (m_identifier && (m_identifier != other.m_identifier))
        return false;

    if (m_offset != other.m_offset)
        return false;

    if (m_oldStructure != other.m_oldStructure)
        return false;
    ASSERT(m_newStructure == other.m_newStructure);

    return true;
}

bool DeleteByVariant::writesStructures() const
{
    return !!newStructure();
}

template<typename Visitor>
void DeleteByVariant::visitAggregateImpl(Visitor& visitor)
{
    m_identifier.visitAggregate(visitor);
}

DEFINE_VISIT_AGGREGATE(DeleteByVariant);

template<typename Visitor>
void DeleteByVariant::markIfCheap(Visitor& visitor)
{
    if (m_oldStructure)
        m_oldStructure->markIfCheap(visitor);
    if (m_newStructure)
        m_newStructure->markIfCheap(visitor);
}

template void DeleteByVariant::markIfCheap(AbstractSlotVisitor&);
template void DeleteByVariant::markIfCheap(SlotVisitor&);

void DeleteByVariant::dump(PrintStream& out) const
{
    dumpInContext(out, nullptr);
}

bool DeleteByVariant::finalize(VM& vm)
{
    if (!vm.heap.isMarked(m_oldStructure))
        return false;
    if (m_newStructure && !vm.heap.isMarked(m_newStructure))
        return false;
    return true;
}

void DeleteByVariant::dumpInContext(PrintStream& out, DumpContext*) const
{
    out.print("<");
    out.print("id='", m_identifier, "', result=", m_result);
    if (m_oldStructure)
        out.print(", ", *m_oldStructure);
    if (m_newStructure)
        out.print(" -> ", *m_newStructure);
    out.print(", offset = ", offset());
    out.print(">");
}

} // namespace JSC

