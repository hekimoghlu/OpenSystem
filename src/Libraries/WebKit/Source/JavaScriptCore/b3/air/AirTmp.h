/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 2, 2023.
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

#include "B3Bank.h"
#include "FPRInfo.h"
#include "GPRInfo.h"
#include "Reg.h"
#include <wtf/HashMap.h>

namespace JSC { namespace B3 { namespace Air {

class Arg;
class Code;

// A Tmp is a generalization of a register. It can be used to refer to any GPR or FPR. It can also
// be used to refer to an unallocated register (i.e. a temporary). Like many Air classes, we use
// deliberately terse naming since we will have to use this name a lot.

class Tmp {
public:
    constexpr Tmp()
        : m_value(0)
    {
    }

    explicit Tmp(Reg reg)
    {
        if (reg) {
            if (reg.isGPR())
                m_value = encodeGPR(reg.gpr());
            else
                m_value = encodeFPR(reg.fpr());
        } else
            m_value = 0;
    }

    explicit Tmp(const Arg&);

    static Tmp gpTmpForIndex(unsigned index)
    {
        Tmp result;
        result.m_value = encodeGPTmp(index);
        return result;
    }

    static Tmp fpTmpForIndex(unsigned index)
    {
        Tmp result;
        result.m_value = encodeFPTmp(index);
        return result;
    }
    
    static Tmp tmpForIndex(Bank bank, unsigned index)
    {
        if (bank == GP)
            return gpTmpForIndex(index);
        ASSERT(bank == FP);
        return fpTmpForIndex(index);
    }

    explicit operator bool() const { return !!m_value; }
    
    bool isGP() const
    {
        return isEncodedGP(m_value);
    }

    bool isFP() const
    {
        return isEncodedFP(m_value);
    }

    // For null tmps, returns GP.
    Bank bank() const
    {
        return isFP() ? FP : GP;
    }

    bool isGPR() const
    {
        return isEncodedGPR(m_value);
    }

    bool isFPR() const
    {
        return isEncodedFPR(m_value);
    }

    bool isReg() const
    {
        return isGPR() || isFPR();
    }

    GPRReg gpr() const
    {
        return decodeGPR(m_value);
    }

    FPRReg fpr() const
    {
        return decodeFPR(m_value);
    }

    Reg reg() const
    {
        if (isGP())
            return gpr();
        return fpr();
    }

    bool hasTmpIndex() const
    {
        return !isReg();
    }

    unsigned gpTmpIndex() const
    {
        return decodeGPTmp(m_value);
    }

    unsigned fpTmpIndex() const
    {
        return decodeFPTmp(m_value);
    }
    
    unsigned tmpIndex(Bank bank) const
    {
        if (bank == GP)
            return gpTmpIndex();
        ASSERT(bank == FP);
        return fpTmpIndex();
    }

    unsigned tmpIndex() const
    {
        if (isGP())
            return gpTmpIndex();
        return fpTmpIndex();
    }
    
    template<Bank bank> class Indexed;
    template<Bank bank> class AbsolutelyIndexed;
    class LinearlyIndexed;
    
    template<Bank bank>
    Indexed<bank> indexed() const;
    
    template<Bank bank>
    AbsolutelyIndexed<bank> absolutelyIndexed() const;
    
    LinearlyIndexed linearlyIndexed(Code&) const;

    static unsigned indexEnd(Code&, Bank);
    static unsigned absoluteIndexEnd(Code&, Bank);
    static unsigned linearIndexEnd(Code&);
    
    bool isAlive() const
    {
        return !!*this;
    }

    friend bool operator==(const Tmp&, const Tmp&) = default;

    void dump(PrintStream& out) const;

    Tmp(WTF::HashTableDeletedValueType)
        : m_value(std::numeric_limits<int>::max())
    {
    }

    bool isHashTableDeletedValue() const
    {
        return *this == Tmp(WTF::HashTableDeletedValue);
    }

    unsigned hash() const
    {
        return WTF::IntHash<int>::hash(m_value);
    }

    unsigned internalValue() const { return static_cast<unsigned>(m_value); }

    static Tmp tmpForInternalValue(unsigned index)
    {
        Tmp result;
        result.m_value = static_cast<int>(index);
        return result;
    }
    
    static Tmp tmpForAbsoluteIndex(Bank, unsigned);
    
    static Tmp tmpForLinearIndex(Code&, unsigned);
    
private:
    static int encodeGP(unsigned index)
    {
        return 1 + index;
    }

    static int encodeFP(unsigned index)
    {
        return -1 - index;
    }

    static int encodeGPR(GPRReg gpr)
    {
        return encodeGP(gpr - MacroAssembler::firstRegister());
    }

    static int encodeFPR(FPRReg fpr)
    {
        return encodeFP(fpr - MacroAssembler::firstFPRegister());
    }

    static int encodeGPTmp(unsigned index)
    {
        return encodeGPR(MacroAssembler::lastRegister()) + 1 + index;
    }

    static int encodeFPTmp(unsigned index)
    {
        return encodeFPR(MacroAssembler::lastFPRegister()) - 1 - index;
    }

    static bool isEncodedGP(int value)
    {
        return value > 0;
    }

    static bool isEncodedFP(int value)
    {
        return value < 0;
    }

    static bool isEncodedGPR(int value)
    {
        return isEncodedGP(value) && value <= encodeGPR(MacroAssembler::lastRegister());
    }

    static bool isEncodedFPR(int value)
    {
        return isEncodedFP(value) && value >= encodeFPR(MacroAssembler::lastFPRegister());
    }

    static bool isEncodedGPTmp(int value)
    {
        return isEncodedGP(value) && !isEncodedGPR(value);
    }

    static bool isEncodedFPTmp(int value)
    {
        return isEncodedFP(value) && !isEncodedFPR(value);
    }

    static GPRReg decodeGPR(int value)
    {
        ASSERT(isEncodedGPR(value));
        return static_cast<GPRReg>(
            (value - encodeGPR(MacroAssembler::firstRegister())) + MacroAssembler::firstRegister());
    }

    static FPRReg decodeFPR(int value)
    {
        ASSERT(isEncodedFPR(value));
        return static_cast<FPRReg>(
            (encodeFPR(MacroAssembler::firstFPRegister()) - value) +
            MacroAssembler::firstFPRegister());
    }

    static unsigned decodeGPTmp(int value)
    {
        ASSERT(isEncodedGPTmp(value));
        return value - (encodeGPR(MacroAssembler::lastRegister()) + 1);
    }

    static unsigned decodeFPTmp(int value)
    {
        ASSERT(isEncodedFPTmp(value));
        return (encodeFPR(MacroAssembler::lastFPRegister()) - 1) - value;
    }

    // 0: empty Tmp
    // positive: GPRs and then GP temps.
    // negative: FPRs and then FP temps.
    int m_value;
};

struct TmpHash {
    static unsigned hash(const Tmp& key) { return key.hash(); }
    static bool equal(const Tmp& a, const Tmp& b) { return a == b; }
    static constexpr bool safeToCompareToEmptyOrDeleted = true;
};

} } } // namespace JSC::B3::Air

namespace WTF {

template<typename T> struct DefaultHash;
template<> struct DefaultHash<JSC::B3::Air::Tmp> : JSC::B3::Air::TmpHash { };

template<typename T> struct HashTraits;
template<> struct HashTraits<JSC::B3::Air::Tmp> : SimpleClassHashTraits<JSC::B3::Air::Tmp> { };

} // namespace WTF

#endif // ENABLE(B3_JIT)
