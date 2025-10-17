/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 26, 2024.
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
#include "B3Bank.h"

namespace JSC { namespace B3 { namespace Air {

class Code;

class TmpWidth {
public:
    TmpWidth();
    TmpWidth(Code&);
    ~TmpWidth();

    template <Bank bank>
    void recompute(Code&);

    // The width of a Tmp is the number of bits that you need to be able to track without some trivial
    // recovery. A Tmp may have a "subwidth" (say, Width32 on a 64-bit system) if either of the following
    // is true:
    //
    // - The high bits are never read.
    // - The high bits are always zero.
    //
    // This doesn't tell you which of those properties holds, but you can query that using the other
    // methods.
    Width width(Tmp tmp) const
    {
        Widths tmpWidths = widths(tmp);
        return std::min(tmpWidths.use, tmpWidths.def);
    }

    // Return the minimum required width for all defs/uses of this Tmp.
    Width requiredWidth(Tmp tmp)
    {
        Widths tmpWidths = widths(tmp);
        return std::max(tmpWidths.use, tmpWidths.def);
    }

    // This indirectly tells you how much of the tmp's high bits are guaranteed to be zero. The number of
    // high bits that are zero are:
    //
    //     TotalBits - defWidth(tmp)
    //
    // Where TotalBits are the total number of bits in the register, so 64 on a 64-bit system.
    Width defWidth(Tmp tmp) const
    {
        return widths(tmp).def;
    }

    // This tells you how much of Tmp is going to be read.
    Width useWidth(Tmp tmp) const
    {
        return widths(tmp).use;
    }
    
private:
    struct Widths {
        Widths() { }

        Widths(Bank bank)
            : use(minimumWidth(bank))
            , def(minimumWidth(bank))
        {
        }

        Widths(Width useArg, Width defArg)
            : use(useArg)
            , def(defArg)
        {
        }

        void dump(PrintStream& out) const;
        
        Width use;
        Width def;
    };

    inline Widths& widths(Tmp);

    const Widths& widths(Tmp tmp) const
    {
        return const_cast<TmpWidth*>(this)->widths(tmp);
    }

    void addWidths(Tmp tmp, Widths tmpWidths)
    {
        widths(tmp) = tmpWidths;
    }

    Vector<Widths>& widthsVector(Bank bank)
    {
        return bank == GP ? m_widthGP : m_widthFP;
    }

    // These are initialized at the beginning of recompute<bank>, which is called in the constructor for both values of bank.
    Vector<Widths> m_widthGP;
    Vector<Widths> m_widthFP;
};

} } } // namespace JSC::B3::Air

#endif // ENABLE(B3_JIT)
