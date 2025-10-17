/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 3, 2024.
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
#include "DFGDesiredTransitions.h"

#if ENABLE(DFG_JIT)

#include "CodeBlock.h"
#include "DFGCommonData.h"
#include "JSCellInlines.h"
#include "SlotVisitorInlines.h"

namespace JSC { namespace DFG {

DesiredTransition::DesiredTransition(CodeBlock* codeOriginOwner, Structure* oldStructure, Structure* newStructure)
    : m_codeOriginOwner(codeOriginOwner)
    , m_oldStructure(oldStructure)
    , m_newStructure(newStructure)
{
}

template<typename Visitor>
void DesiredTransition::visitChildren(Visitor& visitor)
{
    visitor.appendUnbarriered(m_codeOriginOwner);
    visitor.appendUnbarriered(m_oldStructure);
    visitor.appendUnbarriered(m_newStructure);
}

template void DesiredTransition::visitChildren(AbstractSlotVisitor&);
template void DesiredTransition::visitChildren(SlotVisitor&);

DesiredTransitions::DesiredTransitions(CodeBlock* codeBlock)
    : m_codeBlock(codeBlock)
{
}

DesiredTransitions::~DesiredTransitions() = default;

void DesiredTransitions::addLazily(CodeBlock* codeOriginOwner, Structure* oldStructure, Structure* newStructure)
{
    m_transitions.append(DesiredTransition(codeOriginOwner, oldStructure, newStructure));
}

void DesiredTransitions::reallyAdd(VM& vm, CommonData* common)
{
    FixedVector<WeakReferenceTransition> transitions(m_transitions.size());
    for (unsigned i = 0; i < m_transitions.size(); i++) {
        auto& desiredTransition = m_transitions[i];
        transitions[i] = WeakReferenceTransition(vm, m_codeBlock, desiredTransition.m_codeOriginOwner, desiredTransition.m_oldStructure, desiredTransition.m_newStructure);
    }
    if (!transitions.isEmpty()) {
        ConcurrentJSLocker locker(m_codeBlock->m_lock);
        ASSERT(common->m_transitions.isEmpty());
        common->m_transitions = WTFMove(transitions);
    }
}

template<typename Visitor>
void DesiredTransitions::visitChildren(Visitor& visitor)
{
    for (unsigned i = 0; i < m_transitions.size(); i++)
        m_transitions[i].visitChildren(visitor);
}

template void DesiredTransitions::visitChildren(AbstractSlotVisitor&);
template void DesiredTransitions::visitChildren(SlotVisitor&);

} } // namespace JSC::DFG

#endif // ENABLE(DFG_JIT)
