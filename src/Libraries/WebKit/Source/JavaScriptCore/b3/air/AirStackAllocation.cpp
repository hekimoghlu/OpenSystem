/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 4, 2023.
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
#include "AirStackAllocation.h"

#if ENABLE(B3_JIT)

#include "AirCode.h"
#include "StackAlignment.h"
#include <wtf/ListDump.h>

namespace JSC { namespace B3 { namespace Air {

namespace {

namespace AirStackAllocationInternal {
static constexpr bool verbose = false;
}

template<typename Collection>
void updateFrameSizeBasedOnStackSlotsImpl(Code& code, const Collection& collection)
{
    unsigned frameSize = 0;
    for (StackSlot* slot : collection)
        frameSize = std::max(frameSize, static_cast<unsigned>(-slot->offsetFromFP()));
    code.setFrameSize(WTF::roundUpToMultipleOf<stackAlignmentBytes()>(frameSize) + stackAdjustmentForAlignment());
}

} // anonymous namespace

bool attemptAssignment(
    StackSlot* slot, intptr_t offsetFromFP, const Vector<StackSlot*>& otherSlots)
{
    if (AirStackAllocationInternal::verbose)
        dataLog("Attempting to assign ", pointerDump(slot), " to ", offsetFromFP, " with interference ", pointerListDump(otherSlots), "\n");

    // Need to align it to the slot's desired alignment.
    offsetFromFP = -WTF::roundUpToMultipleOf(slot->alignment(), -offsetFromFP);
    
    for (StackSlot* otherSlot : otherSlots) {
        if (!otherSlot->offsetFromFP())
            continue;
        bool overlap = WTF::rangesOverlap(
            offsetFromFP,
            offsetFromFP + static_cast<intptr_t>(slot->byteSize()),
            otherSlot->offsetFromFP(),
            otherSlot->offsetFromFP() + static_cast<intptr_t>(otherSlot->byteSize()));
        if (overlap)
            return false;
    }

    if (AirStackAllocationInternal::verbose)
        dataLog("Assigned ", pointerDump(slot), " to ", offsetFromFP, "\n");
    slot->setOffsetFromFP(offsetFromFP);
    return true;
}

void assign(StackSlot* slot, const Vector<StackSlot*>& otherSlots)
{
    if (AirStackAllocationInternal::verbose)
        dataLog("Attempting to assign ", pointerDump(slot), " with interference ", pointerListDump(otherSlots), "\n");
    
    if (attemptAssignment(slot, -static_cast<intptr_t>(slot->byteSize()), otherSlots))
        return;

    for (StackSlot* otherSlot : otherSlots) {
        if (!otherSlot->offsetFromFP())
            continue;
        bool didAssign = attemptAssignment(
            slot, otherSlot->offsetFromFP() - static_cast<intptr_t>(slot->byteSize()), otherSlots);
        if (didAssign)
            return;
    }

    RELEASE_ASSERT_NOT_REACHED();
}

Vector<StackSlot*> allocateAndGetEscapedStackSlotsWithoutChangingFrameSize(Code& code)
{
    // Allocate all of the escaped slots in order. This is kind of a crazy algorithm to allow for
    // the possibility of stack slots being assigned frame offsets before we even get here.
    RELEASE_ASSERT(code.frameSize() == stackAdjustmentForAlignment());
    Vector<StackSlot*> assignedEscapedStackSlots;
    Vector<StackSlot*> escapedStackSlotsWorklist;
    for (StackSlot* slot : code.stackSlots()) {
        if (slot->isLocked()) {
            if (slot->offsetFromFP())
                assignedEscapedStackSlots.append(slot);
            else
                escapedStackSlotsWorklist.append(slot);
        } else {
            // It would be super strange to have an unlocked stack slot that has an offset already.
            ASSERT(!slot->offsetFromFP());
        }
    }
    // This is a fairly expensive loop, but it's OK because we'll usually only have a handful of
    // escaped stack slots.
    while (!escapedStackSlotsWorklist.isEmpty()) {
        StackSlot* slot = escapedStackSlotsWorklist.takeLast();
        assign(slot, assignedEscapedStackSlots);
        assignedEscapedStackSlots.append(slot);
    }
    return assignedEscapedStackSlots;
}

void allocateEscapedStackSlots(Code& code)
{
    updateFrameSizeBasedOnStackSlotsImpl(
        code, allocateAndGetEscapedStackSlotsWithoutChangingFrameSize(code));
}

void updateFrameSizeBasedOnStackSlots(Code& code)
{
    updateFrameSizeBasedOnStackSlotsImpl(code, code.stackSlots());
}

} } } // namespace JSC::B3::Air

#endif // ENABLE(B3_JIT)

