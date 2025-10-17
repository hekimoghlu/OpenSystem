/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 11, 2025.
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
#include "MarkingConstraint.h"

#include "JSCInlines.h"
#include "VisitCounter.h"
#include <wtf/TZoneMallocInlines.h>

namespace JSC {

static constexpr bool verboseMarkingConstraint = false;

WTF_MAKE_TZONE_ALLOCATED_IMPL(MarkingConstraint);

MarkingConstraint::MarkingConstraint(CString abbreviatedName, CString name, ConstraintVolatility volatility, ConstraintConcurrency concurrency, ConstraintParallelism parallelism)
    : m_abbreviatedName(abbreviatedName)
    , m_name(WTFMove(name))
    , m_volatility(volatility)
    , m_concurrency(concurrency)
    , m_parallelism(parallelism)
{
}

MarkingConstraint::~MarkingConstraint() = default;

void MarkingConstraint::resetStats()
{
    m_lastVisitCount = 0;
}

void MarkingConstraint::execute(SlotVisitor& visitor)
{
    ASSERT(!visitor.heap()->isMarkingForGCVerifier());
    VisitCounter visitCounter(visitor);
    executeImpl(visitor);
    m_lastVisitCount += visitCounter.visitCount();
    if (verboseMarkingConstraint && visitCounter.visitCount())
        dataLog("(", abbreviatedName(), " visited ", visitCounter.visitCount(), " in execute)");
}

void MarkingConstraint::executeSynchronously(AbstractSlotVisitor& visitor)
{
    prepareToExecuteImpl(NoLockingNecessary, visitor);
    executeImpl(visitor);
}

double MarkingConstraint::quickWorkEstimate(SlotVisitor&)
{
    return 0;
}

double MarkingConstraint::workEstimate(SlotVisitor& visitor)
{
    return lastVisitCount() + quickWorkEstimate(visitor);
}

void MarkingConstraint::prepareToExecute(const AbstractLocker& constraintSolvingLocker, SlotVisitor& visitor)
{
    ASSERT(!visitor.heap()->isMarkingForGCVerifier());
    dataLogIf(Options::logGC(), abbreviatedName());
    VisitCounter visitCounter(visitor);
    prepareToExecuteImpl(constraintSolvingLocker, visitor);
    m_lastVisitCount = visitCounter.visitCount();
    if (verboseMarkingConstraint && visitCounter.visitCount())
        dataLog("(", abbreviatedName(), " visited ", visitCounter.visitCount(), " in prepareToExecute)");
}

void MarkingConstraint::doParallelWork(SlotVisitor& visitor, SharedTask<void(SlotVisitor&)>& task)
{
    ASSERT(!visitor.heap()->isMarkingForGCVerifier());
    VisitCounter visitCounter(visitor);
    task.run(visitor);
    if (verboseMarkingConstraint && visitCounter.visitCount())
        dataLog("(", abbreviatedName(), " visited ", visitCounter.visitCount(), " in doParallelWork)");
    {
        Locker locker { m_lock };
        m_lastVisitCount += visitCounter.visitCount();
    }
}

void MarkingConstraint::prepareToExecuteImpl(const AbstractLocker&, AbstractSlotVisitor&)
{
}

} // namespace JSC

