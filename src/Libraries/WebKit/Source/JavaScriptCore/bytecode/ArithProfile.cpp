/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 27, 2025.
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
#include "ArithProfile.h"

#include "CCallHelpers.h"
#include "JSCJSValueInlines.h"

namespace JSC {

#if ENABLE(JIT)
template<typename BitfieldType>
void ArithProfile<BitfieldType>::emitObserveResult(CCallHelpers& jit, JSValueRegs regs, GPRReg tempGPR, TagRegistersMode mode)
{
    if (!shouldEmitSetDouble() && !shouldEmitSetNonNumeric() && !shouldEmitSetHeapBigInt() && !shouldEmitSetBigInt32())
        return;

    CCallHelpers::JumpList done;
    CCallHelpers::JumpList nonNumeric;

    done.append(jit.branchIfInt32(regs, mode));
    CCallHelpers::Jump notDouble = jit.branchIfNotDoubleKnownNotInt32(regs, mode);
    emitSetDouble(jit, tempGPR);
    done.append(jit.jump());

    notDouble.link(&jit);

#if USE(BIGINT32)
    CCallHelpers::Jump notBigInt32 = jit.branchIfNotBigInt32(regs, tempGPR, mode);
    emitSetBigInt32(jit);
    done.append(jit.jump());
    notBigInt32.link(&jit);
#else
    UNUSED_PARAM(tempGPR);
#endif

    nonNumeric.append(jit.branchIfNotCell(regs, mode));
    nonNumeric.append(jit.branchIfNotHeapBigInt(regs.payloadGPR()));
    emitSetHeapBigInt(jit);
    done.append(jit.jump());

    nonNumeric.link(&jit);
    emitSetNonNumeric(jit);

    done.link(&jit);
}

template<typename BitfieldType>
bool ArithProfile<BitfieldType>::shouldEmitSetDouble() const
{
    BitfieldType mask = ObservedResults::Int32Overflow | ObservedResults::Int52Overflow | ObservedResults::NegZeroDouble | ObservedResults::NonNegZeroDouble;
    return (m_bits & mask) != mask;
}

template<typename BitfieldType>
void ArithProfile<BitfieldType>::emitSetDouble(CCallHelpers& jit, GPRReg scratchGPR) const
{
    if (shouldEmitSetDouble()) {
#if CPU(ARM64)
        jit.move(CCallHelpers::TrustedImm32(ObservedResults::Int32Overflow | ObservedResults::Int52Overflow | ObservedResults::NegZeroDouble | ObservedResults::NonNegZeroDouble), scratchGPR);
        emitUnconditionalSet(jit, scratchGPR);
#else
        UNUSED_PARAM(scratchGPR);
        emitUnconditionalSet(jit, ObservedResults::Int32Overflow | ObservedResults::Int52Overflow | ObservedResults::NegZeroDouble | ObservedResults::NonNegZeroDouble);
#endif
    }
}

template<typename BitfieldType>
bool ArithProfile<BitfieldType>::shouldEmitSetNonNumeric() const
{
    BitfieldType mask = ObservedResults::NonNumeric;
    return (m_bits & mask) != mask;
}

template<typename BitfieldType>
void ArithProfile<BitfieldType>::emitSetNonNumeric(CCallHelpers& jit) const
{
    if (shouldEmitSetNonNumeric())
        emitUnconditionalSet(jit, ObservedResults::NonNumeric);
}

template<typename BitfieldType>
bool ArithProfile<BitfieldType>::shouldEmitSetBigInt32() const
{
#if USE(BIGINT32)
    BitfieldType mask = ObservedResults::BigInt32;
    return (m_bits & mask) != mask;
#else
    return false;
#endif
}

template<typename BitfieldType>
bool ArithProfile<BitfieldType>::shouldEmitSetHeapBigInt() const
{
    BitfieldType mask = ObservedResults::HeapBigInt;
    return (m_bits & mask) != mask;
}

template<typename BitfieldType>
void ArithProfile<BitfieldType>::emitSetHeapBigInt(CCallHelpers& jit) const
{
    if (shouldEmitSetHeapBigInt())
        emitUnconditionalSet(jit, ObservedResults::HeapBigInt);
}

#if USE(BIGINT32)
template<typename BitfieldType>
void ArithProfile<BitfieldType>::emitSetBigInt32(CCallHelpers& jit) const
{
    if (shouldEmitSetBigInt32())
        emitUnconditionalSet(jit, ObservedResults::BigInt32);
}
#endif

template<typename BitfieldType>
void ArithProfile<BitfieldType>::emitUnconditionalSet(CCallHelpers& jit, BitfieldType mask) const
{
    static_assert(std::is_same<BitfieldType, uint16_t>::value);
    jit.or16(CCallHelpers::TrustedImm32(mask), CCallHelpers::AbsoluteAddress(addressOfBits()));
}

template<typename BitfieldType>
void ArithProfile<BitfieldType>::emitUnconditionalSet(CCallHelpers& jit, GPRReg mask) const
{
    jit.or16(mask, CCallHelpers::AbsoluteAddress(addressOfBits()));
}

// Generate the implementations of the functions above for UnaryArithProfile/BinaryArithProfile
// If changing the size of either, add the corresponding lines here.
template class ArithProfile<uint16_t>;
#endif // ENABLE(JIT)

} // namespace JSC

namespace WTF {
    
using namespace JSC;

template <typename T>
void printInternal(PrintStream& out, const ArithProfile<T>& profile)
{
    const char* separator = "";

    out.print("Result:<");
    if (!profile.didObserveNonInt32()) {
        out.print("Int32");
        separator = "|";
    } else {
        if (profile.didObserveNegZeroDouble()) {
            out.print(separator, "NegZeroDouble");
            separator = "|";
        }
        if (profile.didObserveNonNegZeroDouble()) {
            out.print(separator, "NonNegZeroDouble");
            separator = "|";
        }
        if (profile.didObserveNonNumeric()) {
            out.print(separator, "NonNumeric");
            separator = "|";
        }
        if (profile.didObserveInt32Overflow()) {
            out.print(separator, "Int32Overflow");
            separator = "|";
        }
        if (profile.didObserveInt52Overflow()) {
            out.print(separator, "Int52Overflow");
            separator = "|";
        }
        if (profile.didObserveHeapBigInt()) {
            out.print(separator, "HeapBigInt");
            separator = "|";
        }
        if (profile.didObserveBigInt32()) {
            out.print(separator, "BigInt32");
            separator = "|";
        }
    }
    out.print(">");
}

void printInternal(PrintStream& out, const UnaryArithProfile& profile)
{
    printInternal(out, static_cast<ArithProfile<UnaryArithProfileBase>>(profile));

    out.print(" Arg ObservedType:<");
    out.print(profile.argObservedType());
    out.print(">");
}

void printInternal(PrintStream& out, const BinaryArithProfile& profile)
{
    printInternal(out, static_cast<ArithProfile<UnaryArithProfileBase>>(profile));

    if (profile.tookSpecialFastPath())
        out.print(" Took special fast path.");

    out.print(" LHS ObservedType:<");
    out.print(profile.lhsObservedType());
    out.print("> RHS ObservedType:<");
    out.print(profile.rhsObservedType());
    out.print(">");
}

void printInternal(PrintStream& out, const JSC::ObservedType& observedType)
{
    const char* separator = "";
    if (observedType.sawInt32()) {
        out.print(separator, "Int32");
        separator = "|";
    }
    if (observedType.sawNumber()) {
        out.print(separator, "Number");
        separator = "|";
    }
    if (observedType.sawNonNumber()) {
        out.print(separator, "NonNumber");
        separator = "|";
    }
}

} // namespace WTF
