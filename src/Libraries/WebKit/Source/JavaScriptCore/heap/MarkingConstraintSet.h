/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 3, 2024.
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

#include "MarkingConstraint.h"
#include "MarkingConstraintExecutorPair.h"
#include <wtf/BitVector.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/Vector.h>

namespace JSC {

class Heap;
class MarkingConstraintSolver;

class MarkingConstraintSet {
    WTF_MAKE_TZONE_ALLOCATED(MarkingConstraintSet);
    WTF_MAKE_NONCOPYABLE(MarkingConstraintSet);
public:
    MarkingConstraintSet(Heap&);
    ~MarkingConstraintSet();
    
    void didStartMarking();
    
    void add(
        CString abbreviatedName,
        CString name,
        MarkingConstraintExecutorPair&&,
        ConstraintVolatility,
        ConstraintConcurrency = ConstraintConcurrency::Concurrent,
        ConstraintParallelism = ConstraintParallelism::Sequential);
    
    void add(
        CString abbreviatedName, CString name,
        MarkingConstraintExecutorPair&& executors,
        ConstraintVolatility volatility,
        ConstraintParallelism parallelism)
    {
        add(abbreviatedName, name, WTFMove(executors), volatility, ConstraintConcurrency::Concurrent, parallelism);
    }
    
    void add(std::unique_ptr<MarkingConstraint>);

    // The following functions are only used by the real GC via the MarkingConstraintSolver.
    // Hence, we only need the SlotVisitor version.

    // Assuming that the mark stacks are all empty, this will give you a guess as to whether or
    // not the wavefront is advancing.
    bool isWavefrontAdvancing(SlotVisitor&);
    bool isWavefrontRetreating(SlotVisitor& visitor) { return !isWavefrontAdvancing(visitor); }
    
    // Returns true if this executed all constraints and none of them produced new work. This
    // assumes that you've alraedy visited roots and drained from there.
    bool executeConvergence(SlotVisitor&);

    // This function is only used by the verifier GC via Heap::verifyGC().
    // Hence, we only need the AbstractSlotVisitor version.

    // Simply runs all constraints without any shenanigans.
    void executeAllSynchronously(AbstractSlotVisitor&);

private:
    friend class MarkingConstraintSolver;

    bool executeConvergenceImpl(SlotVisitor&);
    
    JSC::Heap& m_heap;
    BitVector m_unexecutedRoots;
    BitVector m_unexecutedOutgrowths;
    Vector<std::unique_ptr<MarkingConstraint>> m_set;
    Vector<MarkingConstraint*> m_ordered;
    Vector<MarkingConstraint*> m_outgrowths;
    unsigned m_iteration { 1 };
};

} // namespace JSC

