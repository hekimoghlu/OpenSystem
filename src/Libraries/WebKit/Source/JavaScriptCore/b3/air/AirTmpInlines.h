/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 5, 2025.
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

#include "AirArg.h"
#include "AirCode.h"
#include "AirTmp.h"

namespace JSC { namespace B3 { namespace Air {

inline Tmp::Tmp(const Arg& arg)
{
    *this = arg.tmp();
}

// When a Hash structure is too slow or when Sets contains most values, you can
// use direct array addressing with Tmps.
template<Bank bank>
struct AbsoluteTmpMapper;

template<>
struct AbsoluteTmpMapper<GP> {
    static unsigned absoluteIndex(const Tmp& tmp)
    {
        ASSERT(tmp.isGP());
        ASSERT(static_cast<int>(tmp.internalValue()) > 0);
        return tmp.internalValue();
    }

    static unsigned absoluteIndex(unsigned tmpIndex)
    {
        return absoluteIndex(Tmp::gpTmpForIndex(tmpIndex));
    }

    static unsigned lastMachineRegisterIndex()
    {
        return absoluteIndex(Tmp(MacroAssembler::lastRegister()));
    }

    static Tmp tmpFromAbsoluteIndex(unsigned tmpIndex)
    {
        return Tmp::tmpForInternalValue(tmpIndex);
    }
};

template<>
struct AbsoluteTmpMapper<FP> {
    static unsigned absoluteIndex(const Tmp& tmp)
    {
        ASSERT(tmp.isFP());
        ASSERT(static_cast<int>(tmp.internalValue()) < 0);
        return -tmp.internalValue();
    }

    static unsigned absoluteIndex(unsigned tmpIndex)
    {
        return absoluteIndex(Tmp::fpTmpForIndex(tmpIndex));
    }

    static unsigned lastMachineRegisterIndex()
    {
        return absoluteIndex(Tmp(MacroAssembler::lastFPRegister()));
    }

    static Tmp tmpFromAbsoluteIndex(unsigned tmpIndex)
    {
        return Tmp::tmpForInternalValue(-tmpIndex);
    }
};

template<Bank theBank>
class Tmp::Indexed : public Tmp {
public:
    static const char* const dumpPrefix;
    
    Indexed(Tmp tmp)
        : Tmp(tmp)
    {
    }
    
    unsigned index() const
    {
        return tmpIndex(theBank);
    }
};

template<Bank theBank>
class Tmp::AbsolutelyIndexed : public Tmp {
public:
    static const char* const dumpPrefix;

    AbsolutelyIndexed(Tmp tmp)
        : Tmp(tmp)
    {
    }
    
    unsigned index() const
    {
        return AbsoluteTmpMapper<theBank>::absoluteIndex(*this);
    }
};

class Tmp::LinearlyIndexed : public Tmp {
public:
    static const char* const dumpPrefix;
    
    LinearlyIndexed(Tmp tmp, Code& code)
        : Tmp(tmp)
        , m_code(&code)
    {
    }
    
    ALWAYS_INLINE unsigned index() const
    {
        if (isGP())
            return AbsoluteTmpMapper<GP>::absoluteIndex(*this);
        return absoluteIndexEnd(*m_code, GP) + AbsoluteTmpMapper<FP>::absoluteIndex(*this);
    }

private:
    Code* m_code;
};

template<Bank theBank>
inline Tmp::Indexed<theBank> Tmp::indexed() const
{
    return Indexed<theBank>(*this);
}

template<Bank theBank>
inline Tmp::AbsolutelyIndexed<theBank> Tmp::absolutelyIndexed() const
{
    return AbsolutelyIndexed<theBank>(*this);
}

inline Tmp::LinearlyIndexed Tmp::linearlyIndexed(Code& code) const
{
    return LinearlyIndexed(*this, code);
}

ALWAYS_INLINE unsigned Tmp::indexEnd(Code& code, Bank bank)
{
    return code.numTmps(bank);
}

ALWAYS_INLINE unsigned Tmp::absoluteIndexEnd(Code& code, Bank bank)
{
    if (bank == GP)
        return AbsoluteTmpMapper<GP>::absoluteIndex(code.numTmps(GP));
    return AbsoluteTmpMapper<FP>::absoluteIndex(code.numTmps(FP));
}

ALWAYS_INLINE unsigned Tmp::linearIndexEnd(Code& code)
{
    return absoluteIndexEnd(code, GP) + absoluteIndexEnd(code, FP);
}

ALWAYS_INLINE Tmp Tmp::tmpForAbsoluteIndex(Bank bank, unsigned index)
{
    if (bank == GP)
        return AbsoluteTmpMapper<GP>::tmpFromAbsoluteIndex(index);
    ASSERT(bank == FP);
    return AbsoluteTmpMapper<FP>::tmpFromAbsoluteIndex(index);
}

ALWAYS_INLINE Tmp Tmp::tmpForLinearIndex(Code& code, unsigned index)
{
    unsigned gpEnd = absoluteIndexEnd(code, GP);
    if (index < gpEnd)
        return tmpForAbsoluteIndex(GP, index);
    return tmpForAbsoluteIndex(FP, index - gpEnd);
}

} } } // namespace JSC::B3::Air

#endif // ENABLE(B3_JIT)
