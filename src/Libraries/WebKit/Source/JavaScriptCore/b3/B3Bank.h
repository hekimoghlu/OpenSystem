/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 7, 2024.
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

#include "B3Type.h"
#include "Reg.h"

namespace JSC { namespace B3 {

enum Bank : int8_t {
    GP,
    FP
};

static constexpr unsigned numBanks = 2;

template<typename Func>
void forEachBank(const Func& func)
{
    func(GP);
    func(FP);
}

inline Bank bankForType(Type type)
{
    switch (type.kind()) {
    case Void:
    case Tuple:
        ASSERT_NOT_REACHED();
        return GP;
    case Int32:
    case Int64:
        return GP;
    case Float:
    case Double:
    case V128:
        return FP;
    }
    ASSERT_NOT_REACHED();
    return GP;
}

inline Bank bankForReg(Reg reg)
{
    return reg.isFPR() ? FP : GP;
}

inline Width minimumWidth(Bank bank)
{
    return bank == GP ? Width8 : Width32;
}

ALWAYS_INLINE constexpr Width conservativeWidthWithoutVectors(Bank bank)
{
    return bank == FP ? Width64 : widthForBytes(sizeof(CPURegister));
}

ALWAYS_INLINE constexpr Width conservativeWidth(Bank bank)
{
    return bank == FP ? Width128 : widthForBytes(sizeof(CPURegister));
}

ALWAYS_INLINE constexpr unsigned conservativeRegisterBytes(Bank bank)
{
    return bytesForWidth(conservativeWidth(bank));
}

ALWAYS_INLINE constexpr unsigned conservativeRegisterBytesWithoutVectors(Bank bank)
{
    return bytesForWidth(conservativeWidthWithoutVectors(bank));
}

} } // namespace JSC::B3

namespace WTF {

class PrintStream;

void printInternal(PrintStream&, JSC::B3::Bank);

} // namespace WTF

#endif // ENABLE(B3_JIT)

