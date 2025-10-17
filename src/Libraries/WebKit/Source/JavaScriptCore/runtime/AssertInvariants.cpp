/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 11, 2025.
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
#include "AssertInvariants.h"

#include "ArithProfile.h"
#include "BaselineJITCode.h"
#include "CodeBlock.h"
#include "DFGJITCode.h"

WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN

namespace JSC {

void assertInvariants()
{
    // Assertions to match LowLevelInterpreter.asm. If you change any of this code, be
    // prepared to change LowLevelInterpreter.asm as well!!
    {
#if USE(JSVALUE64)
        const ptrdiff_t CallFrameHeaderSlots = 5;
#else // USE(JSVALUE64) // i.e. 32-bit version
        const ptrdiff_t CallFrameHeaderSlots = 4;
#endif
        const ptrdiff_t MachineRegisterSize = sizeof(CPURegister);
        const ptrdiff_t SlotSize = 8;

        static_assert(sizeof(Register) == SlotSize);
        static_assert(CallFrame::headerSizeInRegisters == CallFrameHeaderSlots);

        static_assert(!CallFrame::callerFrameOffset());
        static_assert(CallerFrameAndPC::sizeInRegisters == (MachineRegisterSize * 2) / SlotSize);
        static_assert(CallFrame::returnPCOffset() == CallFrame::callerFrameOffset() + MachineRegisterSize);
        static_assert(static_cast<std::underlying_type_t<CallFrameSlot>>(CallFrameSlot::codeBlock) * sizeof(Register) == CallFrame::returnPCOffset() + MachineRegisterSize);
        static_assert(CallFrameSlot::callee * sizeof(Register) == CallFrameSlot::codeBlock * sizeof(Register) + SlotSize);
        static_assert(CallFrameSlot::argumentCountIncludingThis * sizeof(Register) == CallFrameSlot::callee * sizeof(Register) + SlotSize);
        static_assert(CallFrameSlot::thisArgument * sizeof(Register) == CallFrameSlot::argumentCountIncludingThis * sizeof(Register) + SlotSize);
        static_assert(CallFrame::headerSizeInRegisters == CallFrameSlot::thisArgument);

        static_assert(CallFrame::argumentOffsetIncludingThis(0) == CallFrameSlot::thisArgument);

#if CPU(BIG_ENDIAN)
        static_assert(TagOffset == 0);
        static_assert(PayloadOffset == 4);
#else
        static_assert(TagOffset == 4);
        static_assert(PayloadOffset == 0);
#endif

#if ENABLE(C_LOOP)
        ASSERT(CodeBlock::llintBaselineCalleeSaveSpaceAsVirtualRegisters() == 1);
#elif USE(JSVALUE32_64)
        ASSERT(CodeBlock::llintBaselineCalleeSaveSpaceAsVirtualRegisters() == 1);
#elif CPU(X86_64) || CPU(ARM64)
        ASSERT(CodeBlock::llintBaselineCalleeSaveSpaceAsVirtualRegisters() == 4);
#endif

        ASSERT(!(reinterpret_cast<ptrdiff_t>((reinterpret_cast<WriteBarrier<JSCell>*>(0x4000)->slot())) - 0x4000));
    }

    // FIXME: make these assertions less horrible.
#if ASSERT_ENABLED
    Vector<int> testVector;
    testVector.resize(42);
    ASSERT(std::bit_cast<uint32_t*>(&testVector)[sizeof(void*) / sizeof(uint32_t) + 1] == 42);
    ASSERT(std::bit_cast<int**>(&testVector)[0] == testVector.begin());
#endif

    {
        UnaryArithProfile arithProfile;
        arithProfile.argSawInt32();
        ASSERT(arithProfile.bits() == UnaryArithProfile::observedIntBits());
        ASSERT(arithProfile.argObservedType().isOnlyInt32());
    }
    {
        UnaryArithProfile arithProfile;
        arithProfile.argSawNumber();
        ASSERT(arithProfile.bits() == UnaryArithProfile::observedNumberBits());
        ASSERT(arithProfile.argObservedType().isOnlyNumber());
    }

    {
        BinaryArithProfile arithProfile;
        arithProfile.lhsSawInt32();
        arithProfile.rhsSawInt32();
        ASSERT(arithProfile.bits() == BinaryArithProfile::observedIntIntBits());
        ASSERT(arithProfile.lhsObservedType().isOnlyInt32());
        ASSERT(arithProfile.rhsObservedType().isOnlyInt32());
    }
    {
        BinaryArithProfile arithProfile;
        arithProfile.lhsSawNumber();
        arithProfile.rhsSawInt32();
        ASSERT(arithProfile.bits() == BinaryArithProfile::observedNumberIntBits());
        ASSERT(arithProfile.lhsObservedType().isOnlyNumber());
        ASSERT(arithProfile.rhsObservedType().isOnlyInt32());
    }
    {
        BinaryArithProfile arithProfile;
        arithProfile.lhsSawNumber();
        arithProfile.rhsSawNumber();
        ASSERT(arithProfile.bits() == BinaryArithProfile::observedNumberNumberBits());
        ASSERT(arithProfile.lhsObservedType().isOnlyNumber());
        ASSERT(arithProfile.rhsObservedType().isOnlyNumber());
    }
    {
        BinaryArithProfile arithProfile;
        arithProfile.lhsSawInt32();
        arithProfile.rhsSawNumber();
        ASSERT(arithProfile.bits() == BinaryArithProfile::observedIntNumberBits());
        ASSERT(arithProfile.lhsObservedType().isOnlyInt32());
        ASSERT(arithProfile.rhsObservedType().isOnlyNumber());
    }

#if ENABLE(DFG_JIT)
    // We share the same layout for particular fields in all JITData to make our data IC assume this.
    static_assert(BaselineJITData::offsetOfGlobalObject() == DFG::JITData::offsetOfGlobalObject());
    static_assert(BaselineJITData::offsetOfStackOffset() == DFG::JITData::offsetOfStackOffset());
#endif
}

} // namespace JSC

WTF_ALLOW_UNSAFE_BUFFER_USAGE_END
