/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 5, 2023.
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

#include "CallFrame.h"
#include "JSCJSValue.h"
#include <wtf/MathExtras.h>

namespace JSC {

// NB. Different platforms may have different requirements here. But 16 bytes is very common.
constexpr unsigned stackAlignmentBytes() { return 16; }

constexpr unsigned stackAlignmentRegisters()
{
    return stackAlignmentBytes() / sizeof(EncodedJSValue);
}
static_assert(stackAlignmentRegisters() == 2, "LLInt, CLoop, and JIT rely on this");

// The number of bytes the SP needs to be adjusted downwards to get an aligned SP after a function prologue.
// I.e.: (callFrameRegister - stackAdjustmentForAlignment()) % stackAlignmentBytes() == 0 always;
constexpr unsigned stackAdjustmentForAlignment()
{
    if (constexpr unsigned excess = sizeof(CallerFrameAndPC) % stackAlignmentBytes())
        return stackAlignmentBytes() - excess;
    return 0;
}

// Align argument count taking into account the CallFrameHeaderSize may be
// an "unaligned" count of registers.
constexpr unsigned roundArgumentCountToAlignFrame(unsigned argumentCount)
{
    return WTF::roundUpToMultipleOf(stackAlignmentRegisters(), argumentCount + CallFrame::headerSizeInRegisters) - CallFrame::headerSizeInRegisters;
}

// Align local register count to make the last local end on a stack aligned address given the
// CallFrame is at an address that is stack aligned minus CallerFrameAndPC::sizeInRegisters
constexpr unsigned roundLocalRegisterCountForFramePointerOffset(unsigned localRegisterCount)
{
    return WTF::roundUpToMultipleOf(stackAlignmentRegisters(), localRegisterCount + CallerFrameAndPC::sizeInRegisters) - CallerFrameAndPC::sizeInRegisters;
}

constexpr unsigned argumentCountForStackSize(unsigned sizeInBytes)
{
    unsigned sizeInRegisters = sizeInBytes / sizeof(void*);

    if (sizeInRegisters <= CallFrame::headerSizeInRegisters)
        return 0;

    return sizeInRegisters - CallFrame::headerSizeInRegisters;
}

inline unsigned logStackAlignmentRegisters()
{
    return WTF::fastLog2(stackAlignmentRegisters());
}

} // namespace JSC
