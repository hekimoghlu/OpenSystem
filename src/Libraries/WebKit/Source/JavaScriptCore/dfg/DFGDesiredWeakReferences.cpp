/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 30, 2024.
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
#include "DFGDesiredWeakReferences.h"

#if ENABLE(DFG_JIT)

#include "CodeBlock.h"
#include "DFGCommonData.h"
#include "JSCJSValueInlines.h"
#include "WriteBarrier.h"

namespace JSC { namespace DFG {

DesiredWeakReferences::DesiredWeakReferences()
    : m_codeBlock(nullptr)
{
}

DesiredWeakReferences::DesiredWeakReferences(CodeBlock* codeBlock)
    : m_codeBlock(codeBlock)
{
}

DesiredWeakReferences::~DesiredWeakReferences() = default;

void DesiredWeakReferences::addLazily(JSCell* cell)
{
    if (cell) {
        if (Structure* structure = jsDynamicCast<Structure*>(cell))
            m_structures.add(structure->id());
        else {
            // There are weird relationships in how optimized CodeBlocks
            // point to other CodeBlocks. We don't want to have them be
            // part of the weak pointer set. For example, an optimized CodeBlock
            // having a weak pointer to itself will cause it to get collected.
            RELEASE_ASSERT(!jsDynamicCast<CodeBlock*>(cell));
            m_cells.add(cell);
        }
    }
}

void DesiredWeakReferences::addLazily(JSValue value)
{
    if (value.isCell())
        addLazily(value.asCell());
}

bool DesiredWeakReferences::contains(JSCell* cell)
{
    if (Structure* structure = jsDynamicCast<Structure*>(cell))
        return m_structures.contains(structure->id());
    return m_cells.contains(cell);
}

void DesiredWeakReferences::finalize()
{
    m_finalizedCells = FixedVector<WriteBarrier<JSCell>>(m_cells.size());
    {
        unsigned index = 0;
        for (JSCell* target : m_cells)
            m_finalizedCells[index++].setWithoutWriteBarrier(target);
    }
    m_finalizedStructures = FixedVector<StructureID>(m_structures.size());
    {
        unsigned index = 0;
        for (StructureID structureID : m_structures)
            m_finalizedStructures[index++] = structureID;
    }
}

void DesiredWeakReferences::reallyAdd(VM& vm, CommonData* common)
{
    // We do not emit WriteBarrier here since (1) GC is deferred and (2) we emit write-barrier on CodeBlock when finishing DFG::Plan::reallyAdd.
    ASSERT_UNUSED(vm, vm.heap.isDeferred());
    if (!m_finalizedCells.isEmpty() || !m_finalizedStructures.isEmpty()) {
        ASSERT(common->m_weakStructureReferences.isEmpty());
        ASSERT(common->m_weakReferences.isEmpty());
        // This is just moving a pointer. And we already synchronized with Lock etc. with compiler threads.
        // So at this point, these vectors are fully constructed and baked by the compiler threads.
        // We can just move these pointers to CommonData, and that's enough.
        static_assert(sizeof(m_finalizedStructures) == sizeof(void*));
        static_assert(sizeof(m_finalizedCells) == sizeof(void*));
        common->m_weakStructureReferences = WTFMove(m_finalizedStructures);
        common->m_weakReferences = WTFMove(m_finalizedCells);
    }
}

template<typename Visitor>
void DesiredWeakReferences::visitChildren(Visitor& visitor)
{
    for (JSCell* target : m_cells)
        visitor.appendUnbarriered(target);
    for (StructureID structureID : m_structures)
        visitor.appendUnbarriered(structureID.decode());
}

template void DesiredWeakReferences::visitChildren(AbstractSlotVisitor&);
template void DesiredWeakReferences::visitChildren(SlotVisitor&);

} } // namespace JSC::DFG

#endif // ENABLE(DFG_JIT)
