/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 2, 2021.
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
#include "MarkStackMergingConstraint.h"

#include "GCSegmentedArrayInlines.h"
#include "JSCInlines.h"
#include <wtf/TZoneMallocInlines.h>

namespace JSC {

WTF_MAKE_TZONE_ALLOCATED_IMPL(MarkStackMergingConstraint);

MarkStackMergingConstraint::MarkStackMergingConstraint(JSC::Heap& heap)
    : MarkingConstraint("Msm", "Mark Stack Merging", ConstraintVolatility::GreyedByExecution)
    , m_heap(heap)
{
}

MarkStackMergingConstraint::~MarkStackMergingConstraint() = default;

double MarkStackMergingConstraint::quickWorkEstimate(SlotVisitor&)
{
    return m_heap.m_mutatorMarkStack->size() + m_heap.m_raceMarkStack->size();
}

void MarkStackMergingConstraint::prepareToExecuteImpl(const AbstractLocker&, AbstractSlotVisitor& visitor)
{
    // Logging the work here ensures that the constraint solver knows that it doesn't need to produce
    // anymore work.
    size_t size = m_heap.m_mutatorMarkStack->size() + m_heap.m_raceMarkStack->size();
    visitor.addToVisitCount(size);
    
    dataLogIf(Options::logGC(), "(", size, ")");
}

template<typename Visitor>
void MarkStackMergingConstraint::executeImplImpl(Visitor& visitor)
{
    // We want to skip this constraint for the GC verifier because:
    // 1. There should be no mutator marking action between the End phase and verifyGC().
    //    Hence, we can ignore these stacks.
    // 2. The End phase explicitly calls iterateExecutingAndCompilingCodeBlocks()
    //    to add executing CodeBlocks to m_heap.m_mutatorMarkStack. We want to
    //    leave those unperturbed.
    if (m_heap.m_isMarkingForGCVerifier)
        return;

    m_heap.m_mutatorMarkStack->transferTo(visitor.mutatorMarkStack());
    m_heap.m_raceMarkStack->transferTo(visitor.mutatorMarkStack());
}

void MarkStackMergingConstraint::executeImpl(AbstractSlotVisitor& visitor) { executeImplImpl(visitor); }
void MarkStackMergingConstraint::executeImpl(SlotVisitor& visitor) { executeImplImpl(visitor); }

} // namespace JSC

