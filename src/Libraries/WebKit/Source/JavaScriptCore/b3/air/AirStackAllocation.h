/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 13, 2022.
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

#if ENABLE(B3_JIT)

#include <wtf/Vector.h>

namespace JSC { namespace B3 { namespace Air {

class Code;
class StackSlot;

// This is a collection of utilities shared by both stack allocators
// (allocateStackByGraphColoring and allocateRegistersAndStackByLinearScan).

// Attempt to put the given stack slot at the given offset. Returns false if this would cause
// the slot to overlap with any of the given slots.
bool attemptAssignment(StackSlot*, intptr_t offsetFromFP, const Vector<StackSlot*>& adjacent);

// Performs a first-fit assignment (smallest possible offset) of the given stack slot such that
// it does not overlap with any of the adjacent slots.
void assign(StackSlot*, const Vector<StackSlot*>& adjacent);

// Allocates all stack slots that escape - that is, that don't have live ranges that can be
// determined by looking at their uses. Returns a vector of slots that got assigned offsets.
// This assumes that no stack allocation has happened previously, and so frame size is zero.
Vector<StackSlot*> allocateAndGetEscapedStackSlotsWithoutChangingFrameSize(Code&);

// Same as allocateAndGetEscapedStackSlotsWithoutChangingFrameSize, but does not return the
// assigned slots, and does set the frame size based on the largest extent of any of the
// allocated slots.
void allocateEscapedStackSlots(Code&);

// Updates Code::frameSize based on the largest extent of any stack slot. This is useful to
// call after performing stack allocation.
void updateFrameSizeBasedOnStackSlots(Code&);

} } } // namespace JSC::B3::Air

#endif // ENABLE(B3_JIT)

