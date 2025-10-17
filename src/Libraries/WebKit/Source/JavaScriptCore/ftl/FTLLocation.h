/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 15, 2024.
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

#if ENABLE(FTL_JIT)

#include "DFGCommon.h"
#include "FPRInfo.h"
#include "GPRInfo.h"
#include "Reg.h"
#include <wtf/HashMap.h>

namespace JSC {

namespace B3 {
class ValueRep;
} // namespace B3

namespace FTL {

class Location {
public:
    enum Kind {
        Unprocessed,
        Register,
        Indirect,
        Constant
    };
    
    Location()
        : m_kind(Unprocessed)
    {
        u.constant = 0;
    }
    
    Location(WTF::HashTableDeletedValueType)
        : m_kind(Unprocessed)
    {
        u.constant = 1;
    }

    static Location forRegister(Reg reg, int32_t addend)
    {
        Location result;
        result.m_kind = Register;
        result.u.variable.regIndex = reg.index();
        result.u.variable.offset = addend;
        return result;
    }
    
    static Location forIndirect(Reg reg, int32_t offset)
    {
        Location result;
        result.m_kind = Indirect;
        result.u.variable.regIndex = reg.index();
        result.u.variable.offset = offset;
        return result;
    }
    
    static Location forConstant(int64_t constant)
    {
        Location result;
        result.m_kind = Constant;
        result.u.constant = constant;
        return result;
    }

    static Location forValueRep(const B3::ValueRep&);

    Kind kind() const { return m_kind; }

    bool hasReg() const { return kind() == Register || kind() == Indirect; }
    Reg reg() const
    {
        ASSERT(hasReg());
        return Reg::fromIndex(u.variable.regIndex);
    }
    
    bool hasOffset() const { return kind() == Indirect; }
    int32_t offset() const
    {
        ASSERT(hasOffset());
        return u.variable.offset;
    }
    
    bool hasAddend() const { return kind() == Register; }
    int32_t addend() const
    {
        ASSERT(hasAddend());
        return u.variable.offset;
    }
    
    bool hasConstant() const { return kind() == Constant; }
    int64_t constant() const
    {
        ASSERT(hasConstant());
        return u.constant;
    }
    
    explicit operator bool() const { return kind() != Unprocessed || u.variable.offset; }

    bool operator!() const { return !static_cast<bool>(*this); }
    
    bool isHashTableDeletedValue() const { return kind() == Unprocessed && u.variable.offset; }
    
    bool operator==(const Location& other) const
    {
        return m_kind == other.m_kind
            && u.constant == other.u.constant;
    }
    
    unsigned hash() const
    {
        unsigned result = m_kind;
        
        switch (kind()) {
        case Unprocessed:
            result ^= u.variable.offset;
            break;

        case Register:
            result ^= u.variable.regIndex;
            break;
            
        case Indirect:
            result ^= u.variable.regIndex;
            result ^= u.variable.offset;
            break;
            
        case Constant:
            result ^= WTF::IntHash<int64_t>::hash(u.constant);
            break;
        }
        
        return WTF::IntHash<unsigned>::hash(result);
    }
    
    void dump(PrintStream&) const;
    
    bool isGPR() const;
    bool involvesGPR() const;
    GPRReg gpr() const;
    GPRReg directGPR() const; // Get the GPR and assert that there is no addend.
    
    bool isFPR() const;
    FPRReg fpr() const;
    
    // Assuming that all registers are saved to the savedRegisters buffer according
    // to FTLSaveRestore convention, this loads the value into the given register.
    // The code that this generates isn't exactly super fast. This assumes that FP
    // and SP contain the same values that they would have contained in the original
    // frame, or that you've done one or more canonically formed calls (i.e. can
    // restore the FP by following the call frame linked list numFramesToPop times,
    // and SP can be recovered by popping FP numFramesToPop-1 times and adding 16).
    void restoreInto(MacroAssembler&, char* savedRegisters, GPRReg result, unsigned numFramesToPop = 0) const;
    
private:
    Kind m_kind;
    union {
        int64_t constant;
        struct {
            unsigned regIndex;
            int32_t offset;
        } variable;
    } u;
};

struct LocationHash {
    static unsigned hash(const Location& key) { return key.hash(); }
    static bool equal(const Location& a, const Location& b) { return a == b; }
    static constexpr bool safeToCompareToEmptyOrDeleted = true;
};

} } // namespace JSC::FTL

namespace WTF {

void printInternal(PrintStream&, JSC::FTL::Location::Kind);

template<typename T> struct DefaultHash;
template<> struct DefaultHash<JSC::FTL::Location> : JSC::FTL::LocationHash { };

template<typename T> struct HashTraits;
template<> struct HashTraits<JSC::FTL::Location> : SimpleClassHashTraits<JSC::FTL::Location> { };

} // namespace WTF

#endif // ENABLE(FTL_JIT)
