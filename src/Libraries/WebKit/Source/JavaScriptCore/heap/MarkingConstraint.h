/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 31, 2025.
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
#pragma once

#include "ConstraintConcurrency.h"
#include "ConstraintParallelism.h"
#include "ConstraintVolatility.h"
#include <limits.h>
#include <wtf/Lock.h>
#include <wtf/Noncopyable.h>
#include <wtf/SharedTask.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/text/CString.h>

namespace JSC {

class MarkingConstraintSet;
class AbstractSlotVisitor;
class SlotVisitor;

class MarkingConstraint {
    WTF_MAKE_NONCOPYABLE(MarkingConstraint);
    WTF_MAKE_TZONE_ALLOCATED(MarkingConstraint);
public:
    JS_EXPORT_PRIVATE MarkingConstraint(
        CString abbreviatedName, CString name, ConstraintVolatility,
        ConstraintConcurrency = ConstraintConcurrency::Concurrent,
        ConstraintParallelism = ConstraintParallelism::Sequential);
    
    JS_EXPORT_PRIVATE virtual ~MarkingConstraint();
    
    unsigned index() const { return m_index; }
    
    const char* abbreviatedName() const { return m_abbreviatedName.data(); }
    const char* name() const { return m_name.data(); }
    
    void resetStats();
    
    size_t lastVisitCount() const { return m_lastVisitCount; }
    
    // The following functions are only used by the real GC via the MarkingConstraintSolver.
    // Hence, we only need the SlotVisitor version.
    void execute(SlotVisitor&);

    JS_EXPORT_PRIVATE virtual double quickWorkEstimate(SlotVisitor&);
    
    double workEstimate(SlotVisitor&);
    
    void prepareToExecute(const AbstractLocker& constraintSolvingLocker, SlotVisitor&);
    
    void doParallelWork(SlotVisitor&, SharedTask<void(SlotVisitor&)>&);
    
    ConstraintVolatility volatility() const { return m_volatility; }
    
    ConstraintConcurrency concurrency() const { return m_concurrency; }
    ConstraintParallelism parallelism() const { return m_parallelism; }

protected:
    virtual void executeImpl(AbstractSlotVisitor&) = 0;
    virtual void executeImpl(SlotVisitor&) = 0;
    JS_EXPORT_PRIVATE virtual void prepareToExecuteImpl(const AbstractLocker& constraintSolvingLocker, AbstractSlotVisitor&);

    // This function is only used by the verifier GC via Heap::verifyGC().
    // Hence, we only need the AbstractSlotVisitor version.
    void executeSynchronously(AbstractSlotVisitor&);

private:
    friend class MarkingConstraintSet; // So it can set m_index.
    
    CString m_abbreviatedName;
    CString m_name;
    size_t m_lastVisitCount { 0 };
    unsigned m_index { UINT_MAX };
    ConstraintVolatility m_volatility;
    ConstraintConcurrency m_concurrency;
    ConstraintParallelism m_parallelism;
    Lock m_lock;
};

} // namespace JSC

