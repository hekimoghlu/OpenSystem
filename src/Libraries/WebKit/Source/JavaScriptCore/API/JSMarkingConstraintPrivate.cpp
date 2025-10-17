/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 29, 2025.
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
#include "JSMarkingConstraintPrivate.h"

#include "APICast.h"
#include "SimpleMarkingConstraint.h"

using namespace JSC;

namespace {

Atomic<unsigned> constraintCounter;

struct Marker : JSMarker {
    AbstractSlotVisitor* visitor;
};

bool isMarked(JSMarkerRef markerRef, JSObjectRef objectRef)
{
    if (!objectRef)
        return true; // Null is an immortal object.
    
    return static_cast<Marker*>(markerRef)->visitor->isMarked(toJS(objectRef));
}

void mark(JSMarkerRef markerRef, JSObjectRef objectRef)
{
    if (!objectRef)
        return;
    
    static_cast<Marker*>(markerRef)->visitor->appendHiddenUnbarriered(toJS(objectRef));
}

} // anonymous namespace

void JSContextGroupAddMarkingConstraint(JSContextGroupRef group, JSMarkingConstraint constraintCallback, void *userData)
{
    VM& vm = *toJS(group);
    JSLockHolder locker(vm);
    
    unsigned constraintIndex = constraintCounter.exchangeAdd(1);
    
    // This is a guess. The algorithm should be correct no matter what we pick. This means
    // that we expect this constraint to mark things even during a stop-the-world full GC, but
    // we don't expect it to be able to mark anything at the very start of a GC before anything
    // else gets marked.
    ConstraintVolatility volatility = ConstraintVolatility::GreyedByMarking;
    
    auto constraint = makeUnique<SimpleMarkingConstraint>(
        toCString("Amc", constraintIndex, "(", RawPointer(constraintCallback), ")"),
        toCString("API Marking Constraint #", constraintIndex, " (", RawPointer(constraintCallback), ", ", RawPointer(userData), ")"),
        MAKE_MARKING_CONSTRAINT_EXECUTOR_PAIR(([constraintCallback, userData] (AbstractSlotVisitor& visitor) {
            Marker marker;
            marker.IsMarked = isMarked;
            marker.Mark = mark;
            marker.visitor = &visitor;
            
            constraintCallback(&marker, userData);
        })),
        volatility,
        ConstraintConcurrency::Sequential);
    
    vm.heap.addMarkingConstraint(WTFMove(constraint));
}
